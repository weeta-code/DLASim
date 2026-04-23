#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <random>
#include <string>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <sys/stat.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ---------------------------------------------------------------------------
// Data Structures
// ---------------------------------------------------------------------------

struct Vec2 {
    double x, y;
    Vec2(double x = 0, double y = 0) : x(x), y(y) {}
};

struct CellKey {
    int cx, cy;
    bool operator==(const CellKey& o) const { return cx == o.cx && cy == o.cy; }
};

struct CellKeyHash {
    size_t operator()(const CellKey& k) const {
        size_t h = std::hash<int>{}(k.cx);
        h ^= std::hash<int>{}(k.cy) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct SimStats {
    int particle_count;
    double max_radius;
    double radius_of_gyration;
    double center_of_mass_x, center_of_mass_y;
    double bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y;
    double simulation_time_ms;
    uint64_t seed;
};

// ---------------------------------------------------------------------------
// DLA Simulator
// ---------------------------------------------------------------------------

class DLASimulator {
private:
    std::vector<Vec2> particles_;
    std::unordered_map<CellKey, std::vector<int>, CellKeyHash> grid_;

    // Physical parameters
    double particle_radius_;
    double sticking_dist_;
    double sticking_dist_sq_;
    double cell_size_;

    // Walk parameters
    double fine_step_;
    double spawn_padding_;
    double kill_factor_;

    // State
    double max_radius_;

    // RNG
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> angle_dist_{0.0, 2.0 * M_PI};

    CellKey cellOf(double x, double y) const {
        return {(int)std::floor(x / cell_size_), (int)std::floor(y / cell_size_)};
    }

    void insertParticle(int idx) {
        CellKey c = cellOf(particles_[idx].x, particles_[idx].y);
        grid_[c].push_back(idx);
    }

    bool checkSticking(double x, double y) const {
        CellKey c = cellOf(x, y);
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                CellKey nb = {c.cx + dx, c.cy + dy};
                auto it = grid_.find(nb);
                if (it != grid_.end()) {
                    for (int idx : it->second) {
                        double ddx = x - particles_[idx].x;
                        double ddy = y - particles_[idx].y;
                        if (ddx * ddx + ddy * ddy < sticking_dist_sq_) {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }

    void addParticle() {
        double spawn_r = max_radius_ + spawn_padding_;
        double kill_r  = spawn_r * kill_factor_;

        while (true) {
            // Spawn on circle
            double a = angle_dist_(rng_);
            double x = spawn_r * std::cos(a);
            double y = spawn_r * std::sin(a);

            bool stuck = false;
            while (!stuck) {
                double dist = std::sqrt(x * x + y * y);

                // Kill if escaped
                if (dist > kill_r) break;

                // Adaptive step: circle-hopping when far from cluster
                // The largest safe circle centered on (x,y) that doesn't
                // overlap the cluster region has radius =
                //   min(dist - max_radius_ - sticking_dist_, kill_r - dist)
                double gap_to_cluster = dist - max_radius_ - sticking_dist_;
                double gap_to_kill    = kill_r - dist;
                double safe_radius    = std::min(gap_to_cluster, gap_to_kill);

                if (safe_radius > 4.0 * fine_step_) {
                    // Circle hop: jump to random point on circle of
                    // radius (safe_radius - fine_step_) centered at (x,y).
                    // This is exact for 2D Brownian motion by rotational symmetry.
                    double hop_r = safe_radius - fine_step_;
                    double hop_a = angle_dist_(rng_);
                    x += hop_r * std::cos(hop_a);
                    y += hop_r * std::sin(hop_a);
                } else {
                    // Fine-grained random walk near cluster
                    double walk_a = angle_dist_(rng_);
                    x += fine_step_ * std::cos(walk_a);
                    y += fine_step_ * std::sin(walk_a);
                }

                // Check sticking via spatial hash
                if (checkSticking(x, y)) {
                    int new_idx = (int)particles_.size();
                    particles_.push_back(Vec2(x, y));
                    insertParticle(new_idx);

                    double r = std::sqrt(x * x + y * y);
                    if (r > max_radius_) max_radius_ = r;

                    stuck = true;
                }
            }
            if (stuck) return;
        }
    }

public:
    DLASimulator(uint64_t seed,
                 double particle_radius = 1.0,
                 double spawn_padding = 20.0,
                 double kill_factor = 3.0)
        : particle_radius_(particle_radius),
          sticking_dist_(2.0 * particle_radius),
          sticking_dist_sq_(4.0 * particle_radius * particle_radius),
          cell_size_(2.0 * particle_radius),
          fine_step_(particle_radius),
          spawn_padding_(spawn_padding),
          kill_factor_(kill_factor),
          max_radius_(0.0),
          rng_(seed)
    {
        particles_.reserve(10000);
        // Seed particle at origin
        particles_.push_back(Vec2(0, 0));
        insertParticle(0);
    }

    void simulate(int num_particles, bool verbose = false) {
        for (int i = 0; i < num_particles; ++i) {
            addParticle();
            if (verbose && (i + 1) % 500 == 0) {
                std::cerr << "  [" << (i + 1) << "/" << num_particles
                          << "] radius=" << max_radius_ << "\n";
            }
        }
    }

    SimStats computeStats(uint64_t seed_val, double sim_time_ms) const {
        SimStats s{};
        s.particle_count = (int)particles_.size();
        s.seed = seed_val;
        s.simulation_time_ms = sim_time_ms;

        // Center of mass
        double sx = 0, sy = 0;
        for (auto& p : particles_) { sx += p.x; sy += p.y; }
        s.center_of_mass_x = sx / s.particle_count;
        s.center_of_mass_y = sy / s.particle_count;

        // Radius of gyration
        double sr2 = 0;
        for (auto& p : particles_) {
            double dx = p.x - s.center_of_mass_x;
            double dy = p.y - s.center_of_mass_y;
            sr2 += dx * dx + dy * dy;
        }
        s.radius_of_gyration = std::sqrt(sr2 / s.particle_count);

        s.max_radius = max_radius_;

        // Bounding box
        s.bbox_min_x = s.bbox_min_y =  1e18;
        s.bbox_max_x = s.bbox_max_y = -1e18;
        for (auto& p : particles_) {
            s.bbox_min_x = std::min(s.bbox_min_x, p.x);
            s.bbox_min_y = std::min(s.bbox_min_y, p.y);
            s.bbox_max_x = std::max(s.bbox_max_x, p.x);
            s.bbox_max_y = std::max(s.bbox_max_y, p.y);
        }
        return s;
    }

    // Render cluster as grayscale PNG
    void saveImage(const std::string& path, int img_size) const {
        std::vector<unsigned char> buf(img_size * img_size, 0);

        // Map cluster to image: center the cluster, scale to fit with margin
        double extent = max_radius_ + 10.0;
        double scale  = (img_size * 0.9) / (2.0 * extent);
        double cx = img_size / 2.0;
        double cy = img_size / 2.0;

        for (auto& p : particles_) {
            int px = (int)(cx + p.x * scale);
            int py = (int)(cy + p.y * scale);

            int r = std::max(1, (int)(particle_radius_ * scale));
            for (int dy = -r; dy <= r; ++dy) {
                for (int dx = -r; dx <= r; ++dx) {
                    if (dx * dx + dy * dy <= r * r) {
                        int ix = px + dx, iy = py + dy;
                        if (ix >= 0 && ix < img_size && iy >= 0 && iy < img_size) {
                            buf[iy * img_size + ix] = 255;
                        }
                    }
                }
            }
        }
        stbi_write_png(path.c_str(), img_size, img_size, 1, buf.data(), img_size);
    }

    // Save particle coordinates as binary (for potential future use)
    void saveParticles(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        int n = (int)particles_.size();
        f.write(reinterpret_cast<const char*>(&n), sizeof(n));
        f.write(reinterpret_cast<const char*>(particles_.data()), n * sizeof(Vec2));
    }

    // Save metadata as JSON
    void saveMetadata(const std::string& path, const SimStats& stats) const {
        std::ofstream f(path);
        f << std::fixed << std::setprecision(4);
        f << "{\n";
        f << "  \"particle_count\": " << stats.particle_count << ",\n";
        f << "  \"max_radius\": " << stats.max_radius << ",\n";
        f << "  \"radius_of_gyration\": " << stats.radius_of_gyration << ",\n";
        f << "  \"center_of_mass\": [" << stats.center_of_mass_x
          << ", " << stats.center_of_mass_y << "],\n";
        f << "  \"bounding_box\": [" << stats.bbox_min_x << ", "
          << stats.bbox_min_y << ", " << stats.bbox_max_x << ", "
          << stats.bbox_max_y << "],\n";
        f << "  \"simulation_time_ms\": " << stats.simulation_time_ms << ",\n";
        f << "  \"seed\": " << stats.seed << "\n";
        f << "}\n";
    }

    const std::vector<Vec2>& particles() const { return particles_; }
    double maxRadius() const { return max_radius_; }
};

// ---------------------------------------------------------------------------
// Portable mkdir -p
// ---------------------------------------------------------------------------

static void mkdirs(const std::string& path) {
    std::string acc;
    for (size_t i = 0; i < path.size(); ++i) {
        acc += path[i];
        if (path[i] == '/' || i == path.size() - 1) {
            mkdir(acc.c_str(), 0755);
        }
    }
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

static void printUsage(const char* prog) {
    std::cerr
        << "Off-Lattice DLA Simulator\n\n"
        << "Usage: " << prog << " [options]\n\n"
        << "Options:\n"
        << "  --particles N    Particles per cluster, fixed (default: 5000)\n"
        << "  --min_particles  Min particles (enables random range mode)\n"
        << "  --max_particles  Max particles (enables random range mode)\n"
        << "  --size S         Image size in pixels (default: 512)\n"
        << "  --count C        Number of clusters to generate (default: 1)\n"
        << "  --outdir DIR     Output directory (default: ./output)\n"
        << "  --seed N         Base random seed (default: time-based)\n"
        << "  --prefix STR     Filename prefix (default: \"dla\")\n"
        << "  --verbose        Print per-particle progress\n"
        << "  -h, --help       Show this help\n\n"
        << "If --min_particles and --max_particles are set, each cluster\n"
        << "gets a uniformly random particle count in [min, max].\n"
        << "Otherwise --particles is used as a fixed count.\n";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    int num_particles  = 5000;
    int min_particles  = -1;   // -1 = not set
    int max_particles  = -1;
    int image_size     = 512;
    int count          = 1;
    std::string outdir = "./output";
    std::string prefix = "dla";
    bool verbose = false;
    uint64_t base_seed =
        (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--particles") && i + 1 < argc)
            num_particles = std::stoi(argv[++i]);
        else if (!strcmp(argv[i], "--min_particles") && i + 1 < argc)
            min_particles = std::stoi(argv[++i]);
        else if (!strcmp(argv[i], "--max_particles") && i + 1 < argc)
            max_particles = std::stoi(argv[++i]);
        else if (!strcmp(argv[i], "--size") && i + 1 < argc)
            image_size = std::stoi(argv[++i]);
        else if (!strcmp(argv[i], "--count") && i + 1 < argc)
            count = std::stoi(argv[++i]);
        else if (!strcmp(argv[i], "--outdir") && i + 1 < argc)
            outdir = argv[++i];
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            base_seed = std::stoull(argv[++i]);
        else if (!strcmp(argv[i], "--prefix") && i + 1 < argc)
            prefix = argv[++i];
        else if (!strcmp(argv[i], "--verbose"))
            verbose = true;
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    // Determine if using random particle range
    bool random_particles = (min_particles > 0 && max_particles > 0);
    if (random_particles && min_particles > max_particles) {
        std::cerr << "Error: --min_particles must be <= --max_particles\n";
        return 1;
    }

    // Create output directories
    std::string img_dir  = outdir + "/images";
    std::string meta_dir = outdir + "/metadata";
    std::string pts_dir  = outdir + "/particles";
    mkdirs(img_dir);
    mkdirs(meta_dir);
    mkdirs(pts_dir);

    std::cout << "=== Off-Lattice DLA Simulator ===\n";
    if (random_particles) {
        std::cout << "Particles per cluster : random [" << min_particles
                  << ", " << max_particles << "]\n";
    } else {
        std::cout << "Particles per cluster : " << num_particles << "\n";
    }
    std::cout << "Image size            : " << image_size << "x" << image_size << "\n"
              << "Clusters to generate  : " << count << "\n"
              << "Output directory      : " << outdir << "\n"
              << "Base seed             : " << base_seed << "\n\n";

    // RNG for random particle counts (seeded deterministically)
    std::mt19937_64 count_rng(base_seed ^ 0xDEADBEEF);

    // Summary stats across all runs
    double total_time = 0;

    for (int i = 0; i < count; ++i) {
        uint64_t seed = base_seed + (uint64_t)i;

        // Pick particle count for this cluster
        int this_particles = num_particles;
        if (random_particles) {
            std::uniform_int_distribution<int> pdist(min_particles, max_particles);
            this_particles = pdist(count_rng);
        }

        auto t0 = std::chrono::steady_clock::now();

        DLASimulator sim(seed);
        sim.simulate(this_particles, verbose);

        auto t1 = std::chrono::steady_clock::now();
        double elapsed_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_time += elapsed_ms;

        SimStats stats = sim.computeStats(seed, elapsed_ms);

        // Build filenames with zero-padded index
        std::ostringstream idx_str;
        idx_str << std::setfill('0') << std::setw(5) << i;
        std::string base = prefix + "_" + idx_str.str();

        sim.saveImage(img_dir + "/" + base + ".png", image_size);
        sim.saveMetadata(meta_dir + "/" + base + ".json", stats);
        sim.saveParticles(pts_dir + "/" + base + ".bin");

        std::cout << "[" << (i + 1) << "/" << count << "] "
                  << stats.particle_count << " particles, "
                  << "R=" << std::fixed << std::setprecision(1)
                  << stats.max_radius << ", "
                  << "Rg=" << stats.radius_of_gyration << ", "
                  << elapsed_ms << " ms\n";
    }

    std::cout << "\nTotal time: " << total_time << " ms"
              << " (avg " << total_time / count << " ms/cluster)\n"
              << "Output: " << outdir << "/\n";

    return 0;
}

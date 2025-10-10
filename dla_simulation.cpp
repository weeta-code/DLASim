#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

// STB Image Write - Single header library for writing images
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

struct Particle {
    double x, y;
    Particle(double x = 0, double y = 0) : x(x), y(y) {}
};

class DLA {
private:
    std::vector<Particle> cluster;
    std::mt19937 gen;
    std::uniform_real_distribution<> dis;
    std::uniform_real_distribution<> angle_dis;
    double death_radius;
    double spawn_radius;
    double particle_radius;
    double sticking_distance;
    
public:
    DLA() : gen(std::random_device{}()), 
            dis(0.0, 1.0), 
            angle_dis(0.0, 2.0 * M_PI),
            death_radius(100.0),
            spawn_radius(50.0),
            particle_radius(1.0),
            sticking_distance(2.0 * particle_radius) {
        // Start with a seed particle at origin
        cluster.push_back(Particle(0, 0));
    }
    
    double distance(const Particle& p1, const Particle& p2) {
        double dx = p1.x - p2.x;
        double dy = p1.y - p2.y;
        return std::sqrt(dx * dx + dy * dy);
    }
    
    bool checkSticking(const Particle& p) {
        for (const auto& existing : cluster) {
            if (distance(p, existing) < sticking_distance) {
                return true;
            }
        }
        return false;
    }
    
    void addParticle() {
        while (true) {
            // Spawn particle on spawn circle
            double angle = angle_dis(gen);
            Particle p(spawn_radius * std::cos(angle), 
                      spawn_radius * std::sin(angle));
            
            // Random walk
            while (true) {
                // Move particle randomly
                double step = 0.5;
                double move_angle = angle_dis(gen);
                p.x += step * std::cos(move_angle);
                p.y += step * std::sin(move_angle);
                
                // Check if particle went too far (death circle)
                double dist_from_origin = std::sqrt(p.x * p.x + p.y * p.y);
                if (dist_from_origin > death_radius) {
                    break; // Restart with new particle
                }
                
                // Check if particle sticks to cluster
                if (checkSticking(p)) {
                    cluster.push_back(p);
                    
                    // Update radii based on cluster growth
                    double max_dist = dist_from_origin + 10;
                    if (max_dist > spawn_radius) {
                        spawn_radius = max_dist;
                        death_radius = spawn_radius * 2;
                    }
                    return;
                }
            }
        }
    }
    
    void simulate(int num_particles) {
        std::cout << "Starting DLA simulation with " << num_particles << " particles...\n";
        for (int i = 0; i < num_particles; i++) {
            addParticle();
            if ((i + 1) % 100 == 0) {
                std::cout << "Added " << (i + 1) << " particles\n";
            }
        }
        std::cout << "Simulation complete!\n";
    }
    
    void savePNG(const std::string& filename, int image_size = 800) {
        // Create image buffer
        std::vector<unsigned char> image(image_size * image_size * 3, 0);
        
        // Find bounds
        double min_x = 0, max_x = 0, min_y = 0, max_y = 0;
        for (const auto& p : cluster) {
            min_x = std::min(min_x, p.x);
            max_x = std::max(max_x, p.x);
            min_y = std::min(min_y, p.y);
            max_y = std::max(max_y, p.y);
        }
        
        // Add padding
        double padding = 20;
        min_x -= padding;
        max_x += padding;
        min_y -= padding;
        max_y += padding;
        
        double range = std::max(max_x - min_x, max_y - min_y);
        
        // Draw particles
        for (const auto& p : cluster) {
            // Map particle position to image coordinates
            int px = static_cast<int>((p.x - min_x) / range * image_size);
            int py = static_cast<int>((p.y - min_y) / range * image_size);
            
            // Draw a small circle for each particle
            int radius = std::max(2, static_cast<int>(particle_radius * image_size / range));
            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    if (dx * dx + dy * dy <= radius * radius) {
                        int x = px + dx;
                        int y = py + dy;
                        if (x >= 0 && x < image_size && y >= 0 && y < image_size) {
                            int idx = (y * image_size + x) * 3;
                            // White particles on black background
                            image[idx] = 255;
                            image[idx + 1] = 255;
                            image[idx + 2] = 255;
                        }
                    }
                }
            }
        }
        
        // Save as PNG
        if (stbi_write_png(filename.c_str(), image_size, image_size, 3, 
                          image.data(), image_size * 3)) {
            std::cout << "Image saved as " << filename << "\n";
        } else {
            std::cerr << "Failed to save image\n";
        }
    }
    
    int getClusterSize() const { return cluster.size(); }
};

int main() {
    DLA dla;
    
    // Simulate with 1000 particles (adjust for more/less detail)
    dla.simulate(1000);
    
    std::cout << "Total particles in cluster: " << dla.getClusterSize() << "\n";
    
    // Save the result as PNG
    dla.savePNG("dla_output.png", 800);
    
    return 0;
}
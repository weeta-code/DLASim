CXX      = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra
TARGET   = dla_sim
SRC      = dla_simulation.cpp

all: $(TARGET)

$(TARGET): $(SRC) stb_image_write.h
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)

# Quick test: generate 5 small clusters
test: $(TARGET)
	./$(TARGET) --particles 500 --size 256 --count 5 --outdir test_output --seed 12345 --verbose
	@echo "Test output in test_output/"
	@ls -la test_output/images/

clean:
	rm -f $(TARGET)
	rm -rf test_output

.PHONY: all test clean

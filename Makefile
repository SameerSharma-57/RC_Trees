build: ./bin/program.out
./bin/program.out: ./src/*.c*
	nvcc ./src/program.cu -o ./bin/program.out

build-debug: ./src/*.c*
	nvcc -g ./src/program.cu -o ./bin/program-debug.out

timestamp_build: ./bin/timestamp.out
./bin/timestamp.out: ./src/*.c*
	nvcc ./src/timestamp.cu -o ./bin/timestamp.out

timestamp_run: ./bin/timestamp.out
	./bin/timestamp.out

run: ./bin/program.out
	./bin/program.out

clean: 
	rm -f ./bin/*.out

build_graph: ./src/generate.cu
	nvcc ./src/generate.cu

build_graph_debug:./src/generate.cu
	nvcc -g ./src/generate.cu -o ./bin/generate-debug.out



build_main: ./bin/main.out
./bin/main.out: ./src/*.c*
	nvcc ./src/Graph_generation/main.cu -o ./bin/main.out



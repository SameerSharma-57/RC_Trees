

class CUDA_timer{
    cudaEvent_t _start,_end;
    bool recording;

    public: 
        CUDA_timer(){
            cudaEventCreate(&_start);
            cudaEventCreate(&_end);
            recording=false;
        }

        void start(){
            recording=true;
            cudaEventRecord(_start);
        }

        void stop(){
            recording=false;
            cudaEventRecord(_end);
        }

        bool isRecording(){
            return recording;
        }

        float time_elapsed_milliseconds(){
            float milliseconds = 0;

            cudaEventElapsedTime(&milliseconds, _start, _end);
            return milliseconds;
        }

        float time_elapsed(){
            float milliseconds = this->time_elapsed_milliseconds();
            milliseconds/=1000;
            return milliseconds;
        }
};
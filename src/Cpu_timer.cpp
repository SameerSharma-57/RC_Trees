#include <ctime>
#include <iostream>
using namespace std;

class CPU_timer{
    clock_t _start,_end;
    bool recording=false;

    public: 
        CPU_timer(){
            _start=clock();
            _end=clock();
            recording=false;
        }

        void start(){
            if(recording){
                string error = "could not start when it is still recording";
                cout<<error<<endl;
                throw error;
            }
            _start=clock();
            recording=true;
        }

        void stop(){
            if(!recording){
                string error = "could not call end before start is called";
                cout<<error<<endl;
                throw error;
            }
            _end=clock();
            recording=false;
        }

        bool isRecording(){
            return recording;
        }

        double time_elapsed(){
            double time = double(_end-_start)/double(CLOCKS_PER_SEC);
            return time;
        }
};

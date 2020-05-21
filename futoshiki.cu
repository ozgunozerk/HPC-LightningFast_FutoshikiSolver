#include<iostream>
#include<string>
#include<sstream>
#include<fstream>
#include <cstdio>
#include <vector>
#include <set>

#define SIZE 5 //Matrix size
#define INPUTSIZE 2306451

//THIS IS how many constraints per instance (at max): 60
//THIS IS how many grids: 144000

__device__ bool FindUnassignedLocation(int* matrix, int &row, int &col)
{
    for (row = 0; row < 5; row++)
        for (col = 0; col < 5; col++)
            if (matrix[row * 5 + col] == -2)
                return true;
    return false;
}


__device__ bool isSafe(int* matrix, int row, int col, int num, int * constraints, int constraint_size)
{
    for (int row = 0; row < 5; row++)
        if (matrix[row * 5 + col] == num)
            return false;


    for (int col = 0; col < 5; col++)
        if (matrix[row * 5 + col] == num)
            return false;


    for(long unsigned int i = 0; i < constraint_size; i+=4)
    {
        if(row == constraints[i] && col == constraints[i+1] && matrix[constraints[i+2] * 5 + constraints[i+3]] != -2 && num < matrix[constraints[i+2] * 5 + constraints[i+3]])
            return false;
        else if(row == constraints[i+2] && col == constraints[i+3] && matrix[constraints[i] * 5 + constraints[i+1]] != -2 && matrix[constraints[i] * 5 + constraints[i+1]] < num)
            return false;
    }
    return true;
}

__global__ void GPU_Futoshiki(int* grids, int* constraints, int* constraint_sizes, int* constraint_beginnings)
{

    int tid = threadIdx.x;
    int x_block = blockIdx.x;
    
    volatile __shared__ bool not_found_flag;  // shared flag to stop other threads when solution found
    __shared__ int local_constraints[60];  // this is a dummy size, large enough to fit for all cases
    __shared__ int constraint_size;  // to get how many constraints do we have (item-wise)
    __shared__ int constraint_start;  // to get where we should start from
    __shared__ int local_final[25];  // for storing the final results locally (also used as a temporary storage)

    // initialization of the shared variables
    not_found_flag = true;
    constraint_size = constraint_sizes[x_block] * 4;
    constraint_start = constraint_beginnings[x_block] * 4;

    if(tid < constraint_size)
        local_constraints[tid] = constraints[constraint_start + tid];
        /*
        if(blockIdx.x == 3 && tid == 0)
        {
            for(int yy = 0; yy < constraint_size; yy+=4)
            {
                printf("%d, %d, %d, %d\n", local_constraints[yy]+1, local_constraints[yy+1]+1, local_constraints[yy+2]+1, local_constraints[yy+3]+1);
            }
        }
        */

    
    if(tid < 25)  // we will assign 5 thread to each cell, 1 thread per value in a cell
    {
        local_final[tid] = grids[x_block * 25 + tid];  // saving the grid into shared memory first (local_final used as temporary storage)

        __syncthreads();
    }
    if(tid < 5)  // we will assign 5 thread to each cell, 1 thread per value in a cell
    {
        int futoshiki[25];  // copy per thread

        for(int q = 0; q < 25; q++)  // filling the copy per thread from shared mem
            futoshiki[q] = local_final[q];  // :)
        
        // start solving the futoshiki
        if(futoshiki[tid] == -2)  // if that cell is empty
        {
            int row;
            int col;

            //futoshiki[row * 5 + col] = value;  // if so, change the value of the current cell
            int staque[50];  // stack implementation via int list
            int stack_counter = 0;  // need for stack implementation, consider as bookmark xd
            int toStack;  // value to store what's popped from the stack
            //int kaputt = 0;  // control variable for if there is no solution
            int allowed_value = 0;  // for roll-back mechanism, allowing us to remember which value to try next
            bool dead_end;  // if we need to roll-back or not
            
            while(not_found_flag)
            {
                if(!FindUnassignedLocation(futoshiki, row, col))
                {
                    not_found_flag = false;
                    for(int x = 0; x < 25; x++)
                        local_final[x] = futoshiki[x];
                }
                else
                {
                    //printf("RETURNED row: %d, col: %d\n", row, col);
                    dead_end = true;  // dummy initialization, if it does not turn to false in the for loop below, it means we have a problem :)
                    //printf("banned value: %d,  deadend: %d\n", banned_value, dead_end);
                    for (int num = allowed_value; num < 5 && dead_end; num++)  // try the values
                    {
                        if (isSafe(futoshiki, row, col, num, local_constraints, constraint_size))  // if the value fits, put it
                        {
                            //printf("now chainging row: %d, col: %d, with num: %d\n", row, col, num);
                            dead_end = false;  // set the dead end to false, since we found a new value
                            allowed_value = 0;  // reset this, since we found a new value
                            futoshiki[row * 5 + col] = num;  // a new step in the matrix 
                            toStack = row * 100 + col * 10 + num;
                            staque[stack_counter++] = toStack;  // push this cell and it's value into the stack, in case of we screw things up XD
                        }
                    }
                    if(dead_end) // means we have a dead end in this cell, so we need to roll back
                    {  // by popping once from our stack
                        toStack = staque[--stack_counter];  // pop the stack and store the value
                        allowed_value = toStack % 10 + 1;  // we need to add 1, otherwise we will be trying the same value over and over again
                        col = (toStack/10) % 10;  // get the cell col
                        row = toStack / 100;  // get the cell row
                        futoshiki[row * 5 + col] = -2;  // set this cell to uninitialized again
                    }
                }
            }
            
        }
        __syncthreads();  // wait for all other threads
    }
    if(not_found_flag == false)
        if(tid < 25) // this is put for debugging, remove and merge with above if when debug done
            grids[x_block * 25 + tid] = local_final[tid];  // write the result back to the global gpu memory
}


int main(int argc, char** argv)
{
  
    std::string filename(argv[1]);
    std::ifstream file(filename.c_str());
    std::ifstream scout(filename.c_str());
    
    int no_grids;
    file >> no_grids;

    int dummy;
    scout >> dummy;

    int* grids = new int[no_grids * 25];
    

    int elem0, elem1, elem2, elem3, elem4;
    int pre_cursor = 0;
    int cursor = 0;
    int csize = 0;
    
    std::string file_line;
    std::string scout_line;

    int* constraint_sizes = new int[no_grids];
    int* constraint_beginnings = new int[no_grids];
    
    std::getline(scout, scout_line);//These are for spare lines
    std::getline(scout, scout_line);
    for(int i = 0; i < INPUTSIZE; i++)
    {
        std::getline(scout, scout_line);
        if(scout_line == "-------")
        {
            csize = i - pre_cursor - 5;
            constraint_sizes[cursor] = csize;
            cursor++;
            pre_cursor = i+1;
        }
    }

    int sum = 0;
    int temp_size;

    std::vector<int> constraint_vector;  // we need a dynamic one that can expand
    std::set<int> constraint_set;  // there are multiple copies of the constraints, WHY ARE you torturing us :(

    std::getline(file, file_line);
    for(int i = 0; i < no_grids; i++)
    {
        std::getline(file, file_line);
        for(int j = 0; j < SIZE; j++)
        {
            std::getline(file, file_line);
            std::istringstream iss(file_line);
            iss >> elem0 >> elem1 >> elem2 >> elem3 >> elem4;
            grids[i*25 + j*5 + 0] = elem0 - 1;
            grids[i*25 + j*5 + 1] = elem1 - 1;
            grids[i*25 + j*5 + 2] = elem2 - 1;
            grids[i*25 + j*5 + 3] = elem3 - 1;
            grids[i*25 + j*5 + 4] = elem4 - 1;
        }
        for(int c = 0; c < constraint_sizes[i]; c++)
        {
            std::getline(file, file_line);
            std::istringstream iss(file_line);
            iss >> elem0 >> elem1 >> elem2 >> elem3;
            elem4 = elem0 * 1000 + elem1 * 100 + elem2 * 10 + elem3;
            constraint_set.insert(elem4);
        } 
        temp_size = constraint_set.size();
        constraint_sizes[i] = temp_size;
        constraint_beginnings[i] = sum;
        sum += temp_size;
        
        
        for (std::set<int>::iterator it=constraint_set.begin(); it!=constraint_set.end(); ++it)
        {
            elem4 = *it;
            constraint_vector.push_back(elem4 / 1000 - 1);  // 4th digit
            constraint_vector.push_back((elem4 / 100) % 10 - 1);  // 3rd digit
            constraint_vector.push_back((elem4 / 10) % 10 - 1);  // 2nd digit
            constraint_vector.push_back(elem4 % 10 - 1);  // 1st digit
        }
        constraint_set.clear();

    }


    temp_size = constraint_vector.size();
    int * constraints = new int[temp_size];
    elem4 = 0;  // my favourite dummy int :)
    for (std::vector<int>::iterator it = constraint_vector.begin(); it != constraint_vector.end(); it++)
    {
        constraints[elem4++] = *it;
    }

    int *grids_d, *constraints_d, *constraint_sizes_d, *constraint_beginnings_d;

    float time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //YOUR MEMORY OPERATIONS//Time accordingly
    cudaEventRecord(start, 0);
    cudaMalloc((void**)&grids_d, no_grids * 25 * sizeof(int));
    cudaMemcpy(grids_d, grids, no_grids * 25 * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&constraints_d, temp_size * sizeof(int));
    cudaMemcpy(constraints_d, constraints, temp_size * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&constraint_sizes_d, no_grids * sizeof(int));
    cudaMemcpy(constraint_sizes_d, constraint_sizes, no_grids * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&constraint_beginnings_d, no_grids * sizeof(int));
    cudaMemcpy(constraint_beginnings_d, constraint_beginnings, no_grids * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("GPU Memory preparation duration: %f ms \n", time);
    //YOUR MEMORY OPERATIONS//
    

    //KERNEL CALL//Time accordingly
    cudaEventRecord(start, 0);
    GPU_Futoshiki<<<no_grids, 96>>>(grids_d, constraints_d, constraint_sizes_d, constraint_beginnings_d);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Kernel Duration: %f ms \n", time);
    //KERNEL CALL//


    //YOUR MEMORY OPERARIONS//Time accordingly
    cudaEventRecord(start, 0);
    cudaMemcpy(grids, grids_d, no_grids * 25 * sizeof(int), cudaMemcpyDeviceToHost); // copy the result back to CPU
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("GPU to CPU Data Transfer Duration: %f ms \n", time);
    //YOUR MEMORY OPERARIONS//



    // free cuda mem
    cudaFree(grids_d);
    cudaFree(constraints_d);
    cudaFree(constraint_sizes_d);
    cudaFree(constraint_beginnings_d);
  
    //OUTPUT FILE
    std::ofstream myfile;
    myfile.open("solution.txt");
    myfile << no_grids << "\n" << "-------" << "\n";
    for(int i = 0; i < no_grids; i++)
    {
        for(int j = 0; j < SIZE; j++)
        {
            elem0 = grids[i*25 + j*5 + 0] + 1;
            elem1 = grids[i*25 + j*5 + 1] + 1;
            elem2 = grids[i*25 + j*5 + 2] + 1;
            elem3 = grids[i*25 + j*5 + 3] + 1;
            elem4 = grids[i*25 + j*5 + 4] + 1;
            myfile << elem0 << " " << elem1 << " " << elem2 << " " << elem3 << " " << elem4 << "\n";
        }
        myfile << "-------" << "\n";
    }
    myfile.close();
    //OUTPUT FILE

    
    /*
    cudaFree(grids_d);
    cudaFree(constraints_d);
    cudaFree(constraint_sizes_d);
    cudaFree(constraint_beginnings_d);
    */
    //Deallocate
    delete[] grids;
   
    delete[] constraints;
  
    delete[] constraint_sizes;

    delete[] constraint_beginnings;
}

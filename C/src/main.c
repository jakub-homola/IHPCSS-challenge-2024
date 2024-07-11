/**
 * @file main.f08
 * @brief This file provides you with the original implementation of pagerank.
 * Your challenge is to optimise it using OpenMP and/or MPI.
 * @author Ludovic Capelli (l.capelli@epcc.ed.ac.uk)
 **/
 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

/// The number of vertices in the graph.
#define GRAPH_ORDER 1000
/// Parameters used in pagerank convergence, do not change.
#define DAMPING_FACTOR 0.85
/// The number of seconds to not exceed forthe calculation loop.
#define MAX_TIME 10

/**
 * @brief Indicates which vertices are connected.
 * @details If an edge links vertex A to vertex B, then adjacency_matrix[A][B]
 * will be 1.0. The absence of edge is represented with value 0.0.
 * Redundant edges are still represented with value 1.0.
 */
double adjacency_matrix[GRAPH_ORDER][GRAPH_ORDER];
double max_diff = 0.0;
double min_diff = 1.0;
double total_diff = 0.0;
 
void initialize_graph(void)
{
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        for(int j = 0; j < GRAPH_ORDER; j++)
        {
            adjacency_matrix[i][j] = 0.0;
        }
    }
}

/**
 * @brief Calculates the pagerank of all vertices in the graph.
 * @param pagerank The array in which store the final pageranks.
 */
void calculate_pagerank(double pagerank[])
{
    double initial_rank = 1.0 / GRAPH_ORDER;

    int mpirank, mpisize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

    if(mpisize != 16) MPI_Abort(MPI_COMM_WORLD, 16);

    int chunkrow = mpirank / 4;
    int chunkcol = mpirank % 4;

    MPI_Comm comm_row;
    MPI_Comm comm_col;
    MPI_Comm_split(MPI_COMM_WORLD, chunkrow, chunkcol, &comm_row);
    MPI_Comm_split(MPI_COMM_WORLD, chunkcol, chunkrow, &comm_col);

    int chunk_size = GRAPH_ORDER / 4;
    int my_i_start = chunkrow * chunk_size;
    int my_i_end = (chunkrow+1) * chunk_size;
    int my_j_start = chunkcol * chunk_size;
    int my_j_end = (chunkcol+1) * chunk_size;
    // I assume the GRAPH_ORDER=1000 will always be that way
 
    // Initialise all vertices to 1/n.
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        pagerank[i] = initial_rank;
    }
 
    double damping_value = (1.0 - DAMPING_FACTOR) / GRAPH_ORDER;
    double diff = 1.0;
    size_t iteration = 0;
    double start = omp_get_wtime();
    double elapsed = omp_get_wtime() - start;
    double time_per_iteration = 0;
    double new_pagerank[2*GRAPH_ORDER];
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        new_pagerank[i] = 0.0;
    }
    double helper[2*GRAPH_ORDER];

    double tm_kernel = 0;
    double tm_exchange = 0;
    double tm_break = 0;
    double tm_reduce = 0;

    // If we exceeded the MAX_TIME seconds, we stop. If we typically spend X seconds on an iteration, and we are less than X seconds away from MAX_TIME, we stop.
    while(1)
    {
        double tm_a1 = omp_get_wtime();
        int do_next = (elapsed < MAX_TIME && (elapsed + time_per_iteration) < MAX_TIME);
        MPI_Bcast(&do_next, 1, MPI_INT, 0, MPI_COMM_WORLD);
        double tm_a2 = omp_get_wtime();
        tm_break += tm_a2 - tm_a1;
        if(!do_next) break;

        double iteration_start = omp_get_wtime();
 
        for(int i = my_i_start; i < my_i_end; i++)
        {
            new_pagerank[i] = 0.0;
        }
 
        double tm_b1 = omp_get_wtime();
        #pragma omp parallel for
		for(int i = my_i_start; i < my_i_end; i++)
        {
			for(int j = my_j_start; j < my_j_end; j++)
            {
                int outdegree = 0;
                for(int k = 0; k < GRAPH_ORDER; k++)
                {
                    if (adjacency_matrix[j][k] == 1.0)
                    {
                        outdegree++;
                    }
                }

				if (adjacency_matrix[j][i] == 1.0)
                {
					new_pagerank[i] += pagerank[j] / (double)outdegree;
				}
			}
		}
        double tm_b2 = omp_get_wtime();
        tm_kernel += tm_b2 - tm_b1;

        double tm_c1 = omp_get_wtime();
        MPI_Reduce(new_pagerank + my_i_start, helper + my_i_start, chunk_size, MPI_DOUBLE, MPI_SUM, chunkrow, comm_row);
        memcpy(new_pagerank + my_i_start, helper + my_i_start, chunk_size * sizeof(double));
        // diagonal block now have correct data in new_pagerank + my_i_start
        // for the diagonal block, my_i_start = my_j_start
        // need to bcast vertically
        MPI_Bcast(new_pagerank + my_j_start, chunk_size, MPI_DOUBLE, chunkcol, comm_col);
        // now all have correct data in new_pagerank + my_j_start (input for next interation)
        MPI_Bcast(new_pagerank + my_i_start, chunk_size, MPI_DOUBLE, chunkrow, comm_row);
        // now evveryone has the correct data
        double tm_c2 = omp_get_wtime();
        tm_exchange += tm_c2 - tm_c1;
 
        for(int i = 0; i < GRAPH_ORDER; i++)
        {
            new_pagerank[i] = DAMPING_FACTOR * new_pagerank[i] + damping_value;
        }
 
        diff = 0.0;
        for(int j = my_j_start; j < my_j_end; j++)
        {
            diff += fabs(new_pagerank[j] - pagerank[j]);
        }
 
        for(int i = 0; i < GRAPH_ORDER; i++)
        {
            pagerank[i] = new_pagerank[i];
        }
            
        double pagerank_total = 0.0;
        for(int j = my_j_start; j < my_j_end; j++)
        {
            pagerank_total += pagerank[j];
        }
        double tm_d1 = omp_get_wtime();
        if(chunkrow == 0)
        {
            double inputs[2] = {diff, pagerank_total};
            double outputs[2];
            MPI_Reduce(inputs, outputs, 2, MPI_DOUBLE, MPI_SUM, 0, comm_row);
            diff = outputs[0];
            pagerank_total = outputs[1];
        }
        double tm_d2 = omp_get_wtime();
        tm_reduce += tm_d2 - tm_d1;
        if(mpirank == 0)
        {
            if(fabs(pagerank_total - 1.0) >= 1E-12)
            {
                printf("[ERROR] Iteration %zu: sum of all pageranks is not 1 but %.12f.\n", iteration, pagerank_total);
            }
            max_diff = (max_diff < diff) ? diff : max_diff;
            total_diff += diff;
            min_diff = (min_diff > diff) ? diff : min_diff;
        }
 
		double iteration_end = omp_get_wtime();
		elapsed = omp_get_wtime() - start;
		iteration++;
		time_per_iteration = elapsed / iteration;
    }
    
    if(mpirank == 0)
    {
        printf("%zu iterations achieved in %.2f seconds\n", iteration, elapsed);
        printf("Timers:   %f   %f   %f   %f\n", tm_break, tm_kernel, tm_exchange, tm_reduce);
    }
}

/**
 * @brief Populates the edges in the graph for testing.
 **/
void generate_nice_graph(int rank)
{
    if(rank == 0) printf("Generate a graph for testing purposes (i.e.: a nice and conveniently designed graph :) )\n");
    double start = omp_get_wtime();
    initialize_graph();
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        for(int j = 0; j < GRAPH_ORDER; j++)
        {
            int source = i;
            int destination = j;
            if(i != j)
            {
                adjacency_matrix[source][destination] = 1.0;
            }
        }
    }
    if(rank == 0) printf("%.2f seconds to generate the graph.\n", omp_get_wtime() - start);
}

/**
 * @brief Populates the edges in the graph for the challenge.
 **/
void generate_sneaky_graph(int rank)
{
    if(rank == 0) printf("Generate a graph for the challenge (i.e.: a sneaky graph :P )\n");
    double start = omp_get_wtime();
    initialize_graph();
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        for(int j = 0; j < GRAPH_ORDER - i; j++)
        {
            int source = i;
            int destination = j;
            if(i != j)
            {
                adjacency_matrix[source][destination] = 1.0;
            }
        }
    }
    if(rank == 0) printf("%.2f seconds to generate the graph.\n", omp_get_wtime() - start);
}

int main(int argc, char* argv[])
{

    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank == 0) printf("This program has two graph generators: generate_nice_graph and generate_sneaky_graph. If you intend to submit, your code will be timed on the sneaky graph, remember to try both.\n");

    // Get the time at the very start.
    double start = omp_get_wtime();
    
    generate_sneaky_graph(rank);
 
    /// The array in which each vertex pagerank is stored.
    double pagerank[2*GRAPH_ORDER];
    calculate_pagerank(pagerank);
 
    if(rank == 0)
    {
        // Calculates the sum of all pageranks. It should be 1.0, so it can be used as a quick verification.
        double sum_ranks = 0.0;
        for(int i = 0; i < GRAPH_ORDER; i++)
        {
            if(i % 100 == 0)
            {
                printf("PageRank of vertex %d: %.6f\n", i, pagerank[i]);
            }
            sum_ranks += pagerank[i];
        }
        printf("Sum of all pageranks = %.12f, total diff = %.12f, max diff = %.12f and min diff = %.12f.\n", sum_ranks, total_diff, max_diff, min_diff);
        double end = omp_get_wtime();
    
        printf("Total time taken: %.2f seconds.\n", end - start);
    }

    MPI_Finalize();
 
    return 0;
}

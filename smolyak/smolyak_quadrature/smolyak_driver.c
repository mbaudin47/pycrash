#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include "smolyak.h"
/*
  Some global data.
*/

/******************************************************************************/

int
main (int argc, char *argv[])
/******************************************************************************/
/*
  Purpose:

    SMOLPACK_INTERACTIVE is an interactive test program for SMOLPACK.

  Modified:

    30 April 2007

  Author:

    Knut Petras

  Reference:

    Knut Petras,
    Smolyak Cubature of Given Polynomial Degree with Few Nodes
    for Increasing Dimension,
    Numerische Mathematik,
    Volume 93, Number 4, February 2003, pages 729-753.

  Parameters:

    Commandline parameter, int DIM, the spatial dimension.
    1 <= DIM <= MAXDIM = 40.

    Commandline parameter, int K, the number of integration stages.
    0 <= K <= ?.

    
  Example
  Use in dimension d = 3 and K = 5 integration stages.
    smolpack_driver 3 5
*/
{
  int bs;
  int dimension;
  double exact;
  int fnum;
  int i;
  int j;
  int k_stage;
  int print_stats;
  int q;
  int qmax = 12;
  double quad;
  int seed;
  double sum;
  int size;
  double *nodes;
  double *weights;
  int index;
  FILE * f_weights;
  FILE * f_nodes;

  printf ("\n");
  printf ("SMOLYAK_DRIVER\n");

  if (argc != 3) {
    printf ("\n");
    printf ("Wrong number of arguments : %d\n", argc);
    printf ("smolpack_driver dimension k_stage\n");
    exit (1);
  }

  // Read dimension
  dimension = atoi (argv[1]);
  k_stage = atoi (argv[2]);
  printf ("  Spatial dimension DIM =     %d\n", dimension);
  printf ("  Number of states K =        %d\n", k_stage);
  printf ("\n");

  // Compute weights and nodes
  q = k_stage + dimension;
  size_smolyak (&dimension, &q, &size);
  printf ("  Size = %d\n", size);
  nodes = (double *) calloc (size * dimension + 1, sizeof (double));
  weights = (double *) calloc (size + 1, sizeof (double));
  quad_smolyak (&dimension, &q, nodes, weights);
  // Print weights and nodes
  f_weights = fopen("weights.txt", "w");
  f_nodes = fopen("nodes.txt", "w");
  printf ("  Write weights\n");
  for(i=0; i < size; i++) {
    fprintf (f_weights, "  w[%d] = %.17e\n", i, weights[i]);
  }
  printf ("  Write nodes\n");
  for(i = 0; i < size; i++) {
    for(j = 0; j < dimension; j++) {
      index = dimension * i + j;
      fprintf (f_nodes, "  x[%d, %d] = %.17e\n", i, j, nodes[index]);
    }
  }
  printf ("  Free memory\n");
  free (nodes);
  free (weights);
  fclose(f_weights);
  fclose(f_nodes);
  printf ("  Ok.\n");
  return 0;
}

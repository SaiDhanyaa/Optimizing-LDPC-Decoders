/******************************************************************************

Welcome to GDB Online.
  GDB online is an online compiler and debugger tool for C, C++, Python, PHP, Ruby, 
  C#, OCaml, VB, Perl, Swift, Prolog, Javascript, Pascal, COBOL, HTML, CSS, JS
  Code, Compile, Run and Debug online from anywhere in world.

*******************************************************************************/
#include <stdio.h>

int main()
{
    int hmat[648][1296];
    int i;
    int j;

    FILE* f = fopen("columnIndexR.txt", "r");
    int index;
    if (f == NULL)
        printf("ERROR\n");
    for (i = 0; i < 648; i++)
        for (j = 0; j < 1296; j++)
            hmat[i][j] = 0;
    
    for (i = 0; i < 648; i++)
        for (j = 0; j < 8; j++) {
            fscanf(f, "%d", &index);
            hmat[i][index] = 1;
        }
    fclose(f);
    
    FILE* f1 = fopen("hmatR.txt", "w");
    for (i = 0; i < 648; i++) {
        for (j = 0; j < 1296; j++)
            fprintf(f1, "%d ", hmat[i][j]);
        fprintf(f1, "\n");
    }
    fclose(f1);

    return 0;
}

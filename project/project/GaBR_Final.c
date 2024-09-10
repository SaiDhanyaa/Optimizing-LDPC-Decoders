/* ###########################################################################################################################
## Organization         : The University of Arizona
## File name            : GaB.c
## Language             : C (ANSI)
## Short description    : Gallager-B Hard decision Bit-Flipping algorithm
#### ######################################################################################################################## */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

#define Mnum    648
#define Nnum    1296
#define DCnum   8
#define arrondi(x) ((ceil(x)-x)<(x-floor(x))?(int)ceil(x):(int)floor(x))
#define min(x,y) ((x)<(y)?(x):(y))
#define signf(x) ((x)>=0?0:1)
#define	max(x,y) ((x)<(y)?(y):(x))
#define SQR(A) ((A)*(A))
#define BPSK(x) (1-2*(x))
#define PI 3.1415926536


//#####################################################################################################
void DataPassGB(int *VtoC,int *CtoV,int *Receivedword,int *InterResult,int *Interleaver,int *ColumnDegree,int N,int NbBranch)
{
	int t,numB,n,buf;
	int Global;
	numB=0;
	for (n=0;n<N;n++)
	{
		//Global=(Amplitude)*(1-2*ReceivedSymbol[n]);
		Global=(1-2*Receivedword[n]); 
		//Global=(1-2*(Decide[n] + Receivedword[n])); //Decide[n]^Receivedword[n];
		for (t=0;t<ColumnDegree[n];t++) Global+=(-2)*CtoV[Interleaver[numB+t]]+1;

		for (t=0;t<ColumnDegree[n];t++)
		{
		  buf=Global-((-2)*CtoV[Interleaver[numB+t]]+1);
		  if (buf<0)  VtoC[Interleaver[numB+t]]= 1; //else VtoC[Interleaver[numB+t]]= 1;
		  else if (buf>0) VtoC[Interleaver[numB+t]]= 0; //else VtoC[Interleaver[numB+t]]= 1;
		  else  VtoC[Interleaver[numB+t]]=Receivedword[n];
		}
		numB=numB+ColumnDegree[n];
	}
}
//#####################################################################################################
//#####################################################################################################
void DataPassGBIter0(int *VtoC,int *CtoV,int *Receivedword,int *InterResult,int *Interleaver,int *ColumnDegree,int N,int NbBranch)
{
	int t,numB,n,buf;
	int Global;
	numB=0;
	for (n=0;n<N;n++)
	{
		for (t=0;t<ColumnDegree[n];t++)     VtoC[Interleaver[numB+t]]=Receivedword[n];
		numB=numB+ColumnDegree[n];
	}
}
//##################################################################################################
void CheckPassGB(int *CtoV,int *VtoC,int M,int NbBranch,int *RowDegree)
{
   int t,numB=0,m,signe;
   for (m=0;m<M;m++)
   {
		signe=0;for (t=0;t<RowDegree[m];t++) signe^=VtoC[numB+t];
	    for (t=0;t<RowDegree[m];t++) 	CtoV[numB+t]=signe^VtoC[numB+t];
		numB=numB+RowDegree[m];
   }
}
//#####################################################################################################
void APP_GB(int *Decide,int *CtoV,int *Receivedword,int *Interleaver,int *ColumnDegree,int N,int M,int NbBranch)
{
   	int t,numB,n,buf;
	int Global;
	numB=0;
	for (n=0;n<N;n++)
	{
		Global=(1-2*Receivedword[n]);
		for (t=0;t<ColumnDegree[n];t++) Global+=(-2)*CtoV[Interleaver[numB+t]]+1;
        if(Global>0) Decide[n]= 0;
        else if (Global<0) Decide[n]= 1;
        else  Decide[n]=Receivedword[n];
		numB=numB+ColumnDegree[n];
	}
}
//#####################################################################################################
int ComputeSyndrome(int *Decide,int **Mat,int *RowDegree,int M, int* syndrome)
{
	int Synd,k,l;
	for (k=0;k<M;k++)
	{
		Synd=0;
  //  printf("Decide(%d): ", k);
  //  for (l=0;l<RowDegree[k];l++)
  //    printf("%d ", Decide[Mat[k][l]]);
  //  printf("\n");
  //  printf("columnIndexes:\n");
		for (l=0;l<RowDegree[k];l++) {
      Synd=Synd^Decide[Mat[k][l]];
    //  if (l>0)
    //    printf("%d ", Synd);
    //  printf("%d ", Mat[k][l]);
    }
    syndrome[k] = Synd;
  //  printf("\t\tSynd(%d): %d\n", k, Synd);
		if (Synd==1) break;
	}

	return(1-Synd);
}
//#####################################################################################################
int GaussianElimination_MRB(int *Perm,int **MatOut,int **Mat,int M,int N)
{
	int k,n,m,m1,buf,ind,indColumn,nb,*Index,dep,Rank;

	Index=(int *)calloc(N,sizeof(int));

	// Triangularization
	indColumn=0;nb=0;dep=0;
	for (m=0;m<M;m++)
	{
		if (indColumn==N) { dep=M-m; break; }

		for (ind=m;ind<M;ind++) { if (Mat[ind][indColumn]!=0) break; }
		// If a "1" is found on the column, permutation of rows
		if (ind<M)
		{
			for (n=indColumn;n<N;n++) { buf=Mat[m][n]; Mat[m][n]=Mat[ind][n]; Mat[ind][n]=buf; }
		// bottom of the column ==> 0
			for (m1=m+1;m1<M;m1++)
			{
				if (Mat[m1][indColumn]==1) { for (n=indColumn;n<N;n++) Mat[m1][n]=Mat[m1][n]^Mat[m][n]; }
			}
			Perm[m]=indColumn;
		}
		// else we "mark" the column.
		else { Index[nb++]=indColumn; m--; }

		indColumn++;
	}

	Rank=M-dep;

	for (n=0;n<nb;n++) Perm[Rank+n]=Index[n];

	// Permutation of the matrix
	for (m=0;m<M;m++) { for (n=0;n<N;n++) MatOut[m][n]=Mat[m][Perm[n]]; }

	// Diagonalization
	for (m=0;m<(Rank-1);m++)
	{
		for (n=m+1;n<Rank;n++)
		{
			if (MatOut[m][n]==1) { for (k=n;k<N;k++) MatOut[m][k]=MatOut[n][k]^MatOut[m][k]; }
		}
	}
	free(Index);
	return(Rank);
}

//#####################################################################################################
int main(int argc, char * argv[])
{
  // Variables Declaration
  int iterNum[] = {116, 206, 699, 3369, 14701};

  FILE *f,*f1;
  int Graine,NbIter,nbtestedframes,NBframes;
  float alpha_max, alpha_min,alpha_step,alpha,NbMonteCarlo;
  // ----------------------------------------------------
  // lecture des param de la ligne de commande
  // ----------------------------------------------------
  char *FileName,*FileMatrix,*FileResult,*FileSimu,*name;
  FileName=(char *)malloc(200);
  FileMatrix=(char *)malloc(200);
  FileResult=(char *)malloc(200);
  FileSimu=(char *)malloc(200);
  name=(char *)malloc(200);

  strcpy(FileMatrix, "columnIndexR.txt"); 
  strcpy(FileResult, "hmatRc_Res.txt"); 
  //--------------Simulation input for GaB BF-------------------------
//  NbMonteCarlo=1000000000000;	    // Maximum nb of codewords sent
  NbMonteCarlo=2;	    // Maximum nb of codewords sent
//  NbIter=100; 	            // Maximum nb of iterations
  NbIter=2; 	            // Maximum nb of iterations
  alpha= 0.01;              // Channel probability of error
  NBframes=100;	            // Simulation stops when NBframes in error
  Graine=1;		            // Seed Initialization for Multiple Simulations

    // brkunl
  alpha_max= 0.0600;		    //Channel Crossover Probability Max and Min
//  alpha_max= 0.0200;
  alpha_min= 0.0200;
//  alpha_min= 0.0600;
  alpha_step=0.0100;


  // ----------------------------------------------------
  // Load Matrix
  // ----------------------------------------------------
  int *ColumnDegree,*RowDegree,**Mat;
  int M,N,m,n,k,i,j;
  M=Mnum; N=Nnum;
  ColumnDegree=(int *)calloc(N,sizeof(int));
  RowDegree=(int *)calloc(M,sizeof(int));
  for (m=0;m<M;m++) RowDegree[m]=DCnum;
  Mat=(int **)calloc(M,sizeof(int *));for (m=0;m<M;m++) Mat[m]=(int *)calloc(RowDegree[m],sizeof(int));
  strcpy(FileName,FileMatrix);
  f=fopen(FileName,"r");for (m=0;m<M;m++) { for (k=0;k<RowDegree[m];k++) fscanf(f,"%d",&Mat[m][k]); }fclose(f);
  for (m=0;m<M;m++) { for (k=0;k<RowDegree[m];k++) ColumnDegree[Mat[m][k]]++; }

  printf("Matrix Loaded \n");

  // ----------------------------------------------------
  // Build Graph
  // ----------------------------------------------------
  int NbBranch,**NtoB,*Interleaver,*ind,numColumn,numBranch;
  NbBranch=0; for (m=0;m<M;m++) NbBranch=NbBranch+RowDegree[m];
  NtoB=(int **)calloc(N,sizeof(int *)); for (n=0;n<N;n++) NtoB[n]=(int *)calloc(ColumnDegree[n],sizeof(int));
  Interleaver=(int *)calloc(NbBranch,sizeof(int));
  ind=(int *)calloc(N,sizeof(int));
  numBranch=0;for (m=0;m<M;m++) { for (k=0;k<RowDegree[m];k++) { numColumn=Mat[m][k]; NtoB[numColumn][ind[numColumn]++]=numBranch++; } }
  free(ind);
  numBranch=0;for (n=0;n<N;n++) { for (k=0;k<ColumnDegree[n];k++) Interleaver[numBranch++]=NtoB[n][k]; }

  printf("Graph Build \n");

  // ----------------------------------------------------
  // Decoder
  // ----------------------------------------------------
  int *CtoV,*VtoC,*Codeword,*Receivedword,*Decide,*U,l,kk;
  int iter,numB;
  CtoV=(int *)calloc(NbBranch,sizeof(int));
  VtoC=(int *)calloc(NbBranch,sizeof(int));
  Codeword=(int *)calloc(N,sizeof(int));
  Receivedword=(int *)calloc(N,sizeof(int));
  Decide=(int *)calloc(N,sizeof(int));
  int* syndrome=(int *)calloc(M,sizeof(int));
  U=(int *)calloc(N,sizeof(int));
  srand48(time(0)+Graine*31+113);

  // ----------------------------------------------------
  // Gaussian Elimination for the Encoding Matrix (Full Representation)
  // ----------------------------------------------------
  int **MatFull,**MatG,*PermG;
  int rank;
  MatG=(int **)calloc(M,sizeof(int *));for (m=0;m<M;m++) MatG[m]=(int *)calloc(N,sizeof(int));
  MatFull=(int **)calloc(M,sizeof(int *));for (m=0;m<M;m++) MatFull[m]=(int *)calloc(N,sizeof(int));
  PermG=(int *)calloc(N,sizeof(int)); for (n=0;n<N;n++) PermG[n]=n;
  for (m=0;m<M;m++) { for (k=0;k<RowDegree[m];k++) { MatFull[m][Mat[m][k]]=1; } }
  rank=GaussianElimination_MRB(PermG,MatG,MatFull,M,N);
  //for (m=0;m<N;m++) printf("%d\t",PermG[m]);printf("\n");

  // Variables for Statistics
  int IsCodeword,nb;
  int NiterMoy,NiterMax;
  int Dmin;
  int NbTotalErrors,NbBitError;
  int NbUnDetectedErrors,NbError;
  int *energy;
  energy=(int *)calloc(N,sizeof(int));

  strcpy(FileName,FileResult);
  f=fopen(FileName,"w");
  fprintf(f,"-----------------------------------------------------Gallager B-----------------------------------------------------\n");
  fprintf(f,"alpha\t  NbEr(BER)\t\t\tNbFer(FER)\t\t Nbtested\tIterAver(Itermax)\tNbUndec(Dmin)\n");

  printf("-----------------------------------------------------Gallager B-----------------------------------------------------\n");
  printf("alpha\t  NbEr(BER)\t\t\tNbFer(FER)\t\t Nbtested\tIterAver(Itermax)\tNbUndec(Dmin)\n");

  const int alpha_num = (int)((alpha_max-alpha_min)/alpha_step);
  double time_taken[alpha_num];
  FILE* f2 = fopen("Matrices.txt", "w+");
  for(i=0,alpha=alpha_max;alpha>=alpha_min;alpha-=alpha_step,i++) {

  NiterMoy=0;NiterMax=0;
  Dmin=1e5;
  NbTotalErrors=0;NbBitError=0;
  NbUnDetectedErrors=0;NbError=0;

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  //--------------------------------------------------------------
  for (nb=0,nbtestedframes=0;nb<NbMonteCarlo;nb++)
  {
    //encoding
    //    for (k=0;k<rank;k++) U[k]=0;
    //	for (k=rank;k<N;k++) U[k]=floor(drand48()*2);
    //	for (k=rank-1;k>=0;k--) { for (l=k+1;l<N;l++) U[k]=U[k]^(MatG[k][l]*U[l]); }
    //	for (k=0;k<N;k++) Codeword[PermG[k]]=U[k];
    sprintf(FileSimu, "CodeWord_%d", nb);
    strcat(FileSimu, ".txt");
    // Specify the path to your folder
    char folder_path[200]; 
    sprintf(folder_path, "/home/u4/haniehta/Documents/ece569/project/Codeword/Alpha%d/", i);
    // Concatenate the folder path and file name to form the complete file path
    char file_path[200]; // Adjust the size according to your needs
    snprintf(file_path, sizeof(file_path), "%s%s", folder_path, FileSimu);

    //    f1=fopen(file_path,"w");
    //    if (f1 == NULL) { perror("Error opening file"); printf("Failed to open %s\n", file_path); return 1; } 
    //    for (n=0;n<N;n++) { fprintf(f1,"%d ",Codeword[n]); }fclose(f1);
	
    if (nb < iterNum[i]) {f1=fopen(file_path,"r");} else {f1=fopen("/home/u4/haniehta/Documents/ece569/project/Receivedword/allzero.txt","r");}
    if (f1 == NULL) { perror("Error opening file"); printf("Failed to open %s\n", file_path); return 1;}
    else { for (n=0;n<N;n++) { fscanf(f1,"%d",&Codeword[n]); }}fclose(f1);

    //    printf("Codeword file:%s\n", file_path);
    //	for (n=0;n<N;n++) { printf("%d", Codeword[n]); }
    //    printf("\n");

    // Add Noise
    //    for ( n=0;n<N;n++)  if (drand48()<alpha) Receivedword[n]=1-Codeword[n]; else Receivedword[n]=Codeword[n];
	
	  sprintf(FileSimu, "ReceivedWord_%d", nb);
    strcat(FileSimu, ".txt");
    sprintf(folder_path, "/home/u4/haniehta/Documents/ece569/project/Receivedword/Alpha%d/", i);
    snprintf(file_path, sizeof(file_path), "%s%s", folder_path, FileSimu);    
    //    f1=fopen(file_path,"w");
    //    if (f1 == NULL) { perror("Error opening file"); printf("Failed to open %s\n", file_path); return 1; } 
    //    for (n=0;n<N;n++) { fprintf(f1,"%d ",Receivedword[n]); }fclose(f1);

    if (nb < iterNum[i]) {f1=fopen(file_path,"r");} else {f1=fopen("/home/u4/haniehta/Documents/ece569/project/Receivedword/allzero.txt","r");}
    if (f1 == NULL) { perror("Error opening file"); printf("Failed to open %s\n", file_path); return 1;}
    else { for (n=0;n<N;n++) { fscanf(f1,"%d",&Receivedword[n]); }}fclose(f1);
    //    printf("Receivedword file:%s\n", file_path);
  //  for (n=0;n<N;n++) { printf("%d", Receivedword[n]); }
  //  printf("\n");
  //  printf("Loading Receivedword completed\n");
	//============================================================================
 	// Decoder
	//============================================================================
	for (k=0;k<NbBranch;k++) {CtoV[k]=0;}
	for (k=0;k<N;k++) Decide[k]=Receivedword[k];
    
	for (iter=0;iter<NbIter;iter++)
  {
  //  printf("Alpha (%.2f), Codeword(%d), Iter(%d)\n", alpha, nb, iter);
    fprintf(f2, "Alpha (%.2f), Codeword(%d), Iter(%d)\n", alpha, nb, iter);
    fprintf(f2, "/////////////////////////////////////////////////////////////////\n");
    if(iter==0) DataPassGBIter0(VtoC,CtoV,Receivedword,Decide,Interleaver,ColumnDegree,N,NbBranch);
    else DataPassGB(VtoC,CtoV,Receivedword,Decide,Interleaver,ColumnDegree,N,NbBranch);
    fprintf(f2, "Receivedword from %s\n", file_path);
    for (k=0;k<N;k++) 
      fprintf(f2, "%d", Receivedword[k]);
    fprintf(f2, "\n/////////////////////////////////////////////////////////////////\n");
    fprintf(f2, "VtoC:\n");
    for (k=0;k<NbBranch;k++) 
      fprintf(f2, "%d", VtoC[k]);
    fprintf(f2, "\n/////////////////////////////////////////////////////////////////\n");
    CheckPassGB(CtoV,VtoC,M,NbBranch,RowDegree);
    APP_GB(Decide,CtoV,Receivedword,Interleaver,ColumnDegree,N,M,NbBranch);
    IsCodeword=ComputeSyndrome(Decide,Mat,RowDegree,M, syndrome);
    fprintf(f2, "Decide:\n");
    for (k=0;k<N;k++) 
      fprintf(f2, "%d", Decide[k]);
    fprintf(f2, "\n/////////////////////////////////////////////////////////////////\n");
  //  fprintf(f2,"Index of decide\n");
  //  for (k=0;k<M;k++)
  //  {
  //    for (l=0;l<RowDegree[k];l++) {
  //      fprintf(f2,"%d ", Decide[Mat[k][l]]);
  //    }
  //    fprintf(f2,"\n");
  //  }
  //  fprintf(f2, "\n/////////////////////////////////////////////////////////////////\n");
    fprintf(f2, "Syndrome:\n");
    for (k=0;k<M;k++) {
    //  for (l=0;l<RowDegree[k];l++)
		//			fprintf(f2, "%d ", Decide[Mat[k][l]]);
		//	fprintf(f2, " = ");
      fprintf(f2, "%d", syndrome[k]);
    }
    fprintf(f2, "\n/////////////////////////////////////////////////////////////////\n");
    fprintf(f2, "IsCodeWord:\n");
    fprintf(f2, "%d", IsCodeword);
    fprintf(f2, "\n/////////////////////////////////////////////////////////////////\n");
		if (IsCodeword) break;
  }
	//============================================================================
  	// Compute Statistics
	//============================================================================
      nbtestedframes++;
	  NbError=0;for (k=0;k<N;k++)  if (Decide[k]!=Codeword[k]) NbError++;
	  NbBitError=NbBitError+NbError;
	// Case Divergence
	  if (!IsCodeword)
	  {
		  NiterMoy=NiterMoy+NbIter;
		  NbTotalErrors++;
	  }
	// Case Convergence to Right Codeword
	  if ((IsCodeword)&&(NbError==0)) { NiterMax=max(NiterMax,iter+1); NiterMoy=NiterMoy+(iter+1); }
	// Case Convergence to Wrong Codeword
	  if ((IsCodeword)&&(NbError!=0))
	  {
		  NiterMax=max(NiterMax,iter+1); NiterMoy=NiterMoy+(iter+1);
		  NbTotalErrors++; NbUnDetectedErrors++;
		  Dmin=min(Dmin,NbError);
	  }
	// Stopping Criterion
	 if (NbTotalErrors==NBframes) break;
  }
    printf("%.5f\t  ", alpha);
    printf("%5d(%1.16f)\t",NbBitError,(float)NbBitError/N/nbtestedframes);
    printf("%3d(%1.16f)\t",NbTotalErrors,(float)NbTotalErrors/nbtestedframes);
    printf("%7d\t\t",nbtestedframes);
    printf("%1.2f(%d)\t\t",(float)NiterMoy/nbtestedframes,NiterMax);
    printf("%d(%d)\n",NbUnDetectedErrors,Dmin);

    fprintf(f,"%.5f\t",alpha);
    fprintf(f,"%5d(%1.8f)\t",NbBitError,(float)NbBitError/N/nbtestedframes);
    fprintf(f,"%3d(%1.8f)\t",NbTotalErrors,(float)NbTotalErrors/nbtestedframes);
    fprintf(f,"%7d\t\t",nbtestedframes);
    fprintf(f,"%1.2f(%d)\t\t",(float)NiterMoy/nbtestedframes,NiterMax);
    fprintf(f,"%d(%d)\n",NbUnDetectedErrors,Dmin);

    clock_gettime(CLOCK_MONOTONIC, &end);
    // Calculate the elapsed time in milliseconds
    time_taken[i] = (end.tv_sec - start.tv_sec) * 1000.0; // seconds to milliseconds
    time_taken[i] += (end.tv_nsec - start.tv_nsec) / 1000000.0;
}
  fprintf(f,"\n");
  for (i=0;i<=alpha_num;i++)
    // Print the elapsed time
    fprintf(f,"Execution time for Alpha%d: %f (ms)\n", i, time_taken[i]);
  fclose(f);
  return(0);
}

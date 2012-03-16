import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

class Matrix {

    //Class variables
    private int rows;
    private int cols;
    
    private double[][] data;

    //Constructor to create the empty matrix
    public Matrix(int rowsIn, int colsIn){
	//Iterator variables
	int cCount, rCount;

	rows = rowsIn;
	cols = colsIn; 

	data = new double[rows][cols];	

    }

    //Constructor to fill matrix with a double array
    public Matrix(int rowsIn, int colsIn, double[][] fill){
	//Iterator variables
	int cCount, rCount;

	rows = rowsIn;
	cols = colsIn; 

	data = fill.clone();

    }

    //Constructor to add a single array
    public Matrix(int rowsIn, int colsIn, double[] fill){
	//Iterator variables
	int cCount, rCount;

	rows = rowsIn;
	cols = colsIn; 

	data = new double[rows][cols];

	//If there is only one column, fill the column
	if (cols == 1){
	    for (rCount = 0; rCount < rows; ++rCount){
		data[rCount][0] = fill[rCount];
	    }  
	}
	else{ //Else, fill the first row
	    data[0] = fill;	    
	}
    }

    //Function to return an item
    public double getItem (int row, int col){   
	return data[row][col];
    }

    //Function to set an item
    public double setItem (int row, int col, double value){
	data[row][col] = value;
	return value;
    }

    //Function to return a row
    public Matrix row (int rowIn){

	double[] rowVals = new double[cols];
	
	System.arraycopy(data[rowIn], 0, rowVals, 0, data[rowIn].length);

	return new Matrix(1, cols, rowVals); 
    }

    //Function to return a column
    public Matrix col (int colIn){

	int rCount;
	
	double[] colVals = new double[rows];
	
	for (rCount = 0; rCount < rows; ++rCount){
	    colVals[rCount] = data[rCount][colIn];
	}
	
	return new Matrix(rows, 1, colVals); 
    }	

    //Function to return the matrix as a list
    //One row at a time
    public double[] as_list (){
	int rCount, cCount, counter = 0;
	double[] retVal = new double[cols*rows];
	
	for (rCount = 0; rCount < rows; rCount++){
	    for (cCount = 0; cCount < cols; cCount++){
		retVal[counter] = data[rCount][cCount];
		counter++;
	    }
	} 

	return retVal;
    }

    //Function to return the array as a list
    public String toString(){
	double[] retVal = this.as_list();

	return java.util.Arrays.toString(retVal);
    }

    //Function to create the identity matrix
    public Matrix identity(int rows){
	int rCount;
	Matrix retMatrix = new Matrix(rows, rows);

	for (rCount = 0; rCount < rows; ++rCount){
	    retMatrix.setItem(rCount, rCount, 1);
	}

	return retMatrix;
    }

    //Lays the values along the diagonal
    public Matrix diagonal(double[] vals){
	int rCount;
	Matrix retMatrix = new Matrix(vals.length, vals.length);

	for (rCount = 0; rCount < vals.length; ++rCount){
	    retMatrix.setItem(rCount, rCount, vals[rCount]);
	}

	return retMatrix;
    }

    //Adds to matrices together provided they have the same dimensions
    public Matrix add (Matrix matA, Matrix matB){
	
	int rCount, cCount;
	Matrix matC = null;

	if (checkDimensions(matA, matB)){
	    
	    matC = new Matrix(matA.rows, matB.rows);

	    for (rCount = 0; rCount < matC.rows; rCount++){
		for (cCount = 0; cCount < matC.cols; cCount++){
		    matC.setItem(rCount, cCount, matA.getItem(rCount, cCount) + matB.getItem(rCount, cCount));
		}
	    }	   

	}

	return matC;
    }

    public double scalar_mult(Matrix other) {
  if (cols == 1 && other.cols == 1 && rows == other.rows) {
      double sum = 0.;
      for (int r = 0; r < rows; r++)
          sum += getItem(r, 0)*other.getItem(r, 0);

      return sum;
  }
  else throw new ArithmeticException("Incorrect matrix dimensions");
    }

    public static Matrix mult(Matrix A, double[] B_values) {
  double[][] B_values_fill = new double[B_values.length][1];
  for (int i = 0; i < B_values.length; i++)
      B_values_fill[i][0] = B_values[i];

  Matrix B = new Matrix(B_values.length, 1, B_values_fill);
  B.printMatrix();
  return mult(A, B);
    }

    /*
     * M = Matrix(A.rows,B.cols)
     * for r in xrange(A.rows):
     *     for c in xrange(B.cols):
     *         for k in xrange(A.cols):
     *             M[r,c] += A[r,k]*B[k,c]
     */
    public static Matrix mult(Matrix A, Matrix B) {
  if (A.cols != B.rows)
      throw new ArithmeticException("Incorrect matrix dimensions");

        Matrix result = new Matrix(A.rows, B.cols);
  for (int r = 0; r < A.rows; r++) {
      for (int c = 0; c < B.cols; c++) {
    for (int k = 0; k < A.cols; k++) {
        double left  = A.getItem(r, k);
        double right = B.getItem(k, c);
        double old = result.getItem(r, c);
        result.setItem(r, c, old+left*right);
    }
      }
  }
  return result;
    }

    //Helper function to ensure the dimensions are the same
    public boolean checkDimensions(Matrix matA, Matrix matB){
	if ((matA.cols == matB.cols) && (matA.rows == matB.rows)){
	   return true;
	 }
	else{
	    return false;
	} 
    }

    //A function to print the matrix
    public void printMatrix(){
	int cCount, rCount;
	String currLine;

	for (rCount = 0; rCount < rows; ++rCount){
	    
	    currLine = "["; 
	    
	    for (cCount = 0; cCount < cols; ++cCount){
		currLine += data[rCount][cCount] + " ";
	    }

	    currLine += "]";

	    System.out.println(currLine);
	}

    }    

    /*
     * def is_almost_symmetric(A, ap=1e-6, rp=1e-4):
     *     if A.rows != A.cols: return False
     *     for r in xrange(A.rows-1):
     *         for c in xrange(r):
     *             delta = abs(A[r,c]-A[c,r])
     *             if delta>ap and delta>max(abs(A[r,c]),abs(A[c,r]))*rp:
     *                 return False
     *     return True
     */
    public static boolean is_almost_symmetric(Matrix matrix, double ap, double rp) {
	if (ap <= 0)
	    ap = 0.000001;
	if (rp <= 0)
	    rp = 0.0001;

	if (matrix.rows != matrix.cols)
	    return false;

	for (int r = 0; r < matrix.rows-1; r++) {
	    for (int c = 0; c < r; c++) {
	        double delta = Math.abs(matrix.getItem(r, c) - matrix.getItem(c, r));
		if (delta > ap && delta > Math.max(Math.abs(matrix.getItem(r, c)), Math.abs(matrix.getItem(c, r)))*rp)
		    return false;
	    }
	}
	return true;
    }

    public static boolean is_almost_zero(Matrix matrix) {
      return is_almost_zero(matrix, 0.0000001);
    }

    public static boolean is_almost_zero(Matrix matrix, double ap) {
      return is_almost_zero(matrix, ap, 0.0000001);
    }

    /*
     * def is_almost_zero(A, ap=1e-6, rp=1e-4):
     *     for r in xrange(A.rows):
     *         for c in xrange(A.cols):
     *             delta = abs(A[r,c]-A[c,r])
     *             if delta>ap and delta>max(abs(A[r,c]),abs(A[c,r]))*rp:
     *                 return False
     *     return True
     */
    public static boolean is_almost_zero(Matrix matrix, double ap, double rp) {
	for (int r = 0; r < matrix.rows-1; r++) {
	    for (int c = 0; c < matrix.cols; c++) {
		double delta = Math.abs(matrix.getItem(r, c) - matrix.getItem(c, r));
		if (delta > ap && delta > Math.max(Math.abs(matrix.getItem(r, c)), Math.abs(matrix.getItem(c, r)))*rp)
		    return false;
	    }
	}
	return true;
    }

    public static double norm(Matrix matrix) {
      return norm(matrix, 1);
    }

    /*
     * def norm(A,p=1):
     *   if isinstance(A,(list,tuple)):
     *     return sum(x**p for x in A)**(1.0/p)
     *   elif isinstance(A,Matrix):
     *     if A.rows==1 or A.cols==1:
     *        return sum(norm(A[r,c])**p \
     *          for r in xrange(A.rows) \
     *          for c in xrange(A.cols))**(1.0/p)
     *     elif p==1:
     *       return max([sum(norm(A[r,c]) \
     *         for r in xrange(A.rows)) \
     *         for c in xrange(A.cols)])
     *     else:
     *       raise NotImplementedError
     *   else:
     *     return abs(A)
     */
    public static double norm(Matrix matrix, int p) {
      if (matrix.rows == 1 || matrix.cols == 1) {
        double n = 0.;
        for (int r = 0; r < matrix.rows; r++)
          for (int c = 0; c < matrix.cols; c++)
            n += Math.pow(matrix.getItem(r, c), p);
        return Math.pow(n, (1./((double)p)));
      }
      else if (p == 1) {
        List<Double> norms = new ArrayList<Double>();
        for (int r = 0; r < matrix.rows; r++)
          for (int c = 0; c < matrix.cols; c++)
            norms.add(norm(matrix.getItem(r, c)));
        Collections.sort(norms);
        return norms.get(norms.size()-1);
      }
      else
        throw new RuntimeException("Not implemented");
    }

    public static double norm(double[] list) {
      double[] copy = (double[])list.clone();
      Arrays.sort(copy);
      return copy[copy.length-1];
    }
    
    public static double norm(double val) {
      return Math.abs(val);
    }

    public static void main(String[] args){

	//Using this main class for testing
	Matrix testMatrix = new Matrix(6, 6);
	double testVal = 45.4543243;

	testMatrix.setItem(1, 1, testVal);
	testMatrix.setItem(1, 2, testVal);

	Matrix newMatrix = testMatrix.col(1);

	double[] vals = new double[4];
	vals[0] = 0;
	vals[1] = 1;
	vals[2] = 2;
	vals[3] = 3;

	testMatrix = newMatrix.diagonal(vals);
	newMatrix = newMatrix.identity(4);
	
	Matrix threeMatrix = testMatrix.add(testMatrix, newMatrix);

	threeMatrix.printMatrix();

  double[][] matA_values = {
      { 14, 9,  3  },
      { 2,  11, 15 },
      { 0,  12, 17 },
      { 5,  2,  3  }
  };

  double[][] matB_values = {
      { 12, 25 },
      { 9,  10 },
      { 8,  5  }
  };

  Matrix matA = new Matrix(4, 3, matA_values);
  Matrix matB = new Matrix(3, 2, matB_values);
  mult(matA, matB).printMatrix();

  double[] matC_values = { 2, 3, 4 };
  mult(matA, matC_values).printMatrix();	

  System.out.println(norm(matA));
  System.out.println(norm(matB));
    }
}

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/*
 * def Markovitz(mu, A, r_free):
 * """Assess Markovitz risk/return.
 * Example:
 * >>> cov = Matrix.from_list([[0.04, 0.006,0.02],
 * ...                        [0.006,0.09, 0.06],
 * ...                        [0.02, 0.06, 0.16]])
 * >>> mu = Matrix.from_list([[0.10],[0.12],[0.15]])
 * >>> r_free = 0.05
 * >>> x, ret, risk = Markovitz(mu, cov, r_free)
 * >>> print x
 * [0.556634..., 0.275080..., 0.1682847...]
 * >>> print ret, risk
 * 0.113915... 0.186747...
 * """
 * x = Matrix(A.rows, 1)
 * x = (1/A)*(mu - r_free)
 * x = x/sum(x[r,0] for r in range(x.rows))
 * portfolio = [x[r,0] for r in range(x.rows)]
 * portfolio_return = mu*x
 * portfolio_risk = sqrt(x*(A*x))
 * return portfolio, portfolio_return, portfolio_risk
 */
class Markovitz {
  private final double[] portfolio;
  private final double portfolio_return;
  private final double portfolio_risk;
  
  public Markovitz(Matrix mu, Matrix A, double r_free) {
    Matrix mu_r_free = new Matrix(mu.rowCount(), mu.colCount());
    for (int r = 0; r < mu.rowCount(); r++)
      for (int c = 0; c < mu.colCount(); c++)
        mu_r_free.setItem(r, c, mu.getItem(r, c)-r_free);

    Matrix x = Matrix.mult(Matrix.rDiv(A, 1.), mu_r_free);
    double sum = 0.;
    for (int r = 0; r < x.rowCount(); r++)
      sum += x.getItem(r, 0);
    x = Matrix.mult(x, 1./sum);

    portfolio = new double[x.rowCount()];
    for (int r = 0; r < x.rowCount(); r++)
      portfolio[r] = x.getItem(r, 0);

    portfolio_return = Matrix.scalar_mult(mu, x);
    portfolio_risk = Math.sqrt(Matrix.scalar_mult(x, Matrix.mult(A, x)));
  }

  public double[] getPortfolio() {
    return portfolio;
  }

  public double getReturn() {
    return portfolio_return;
  }

  public double getRisk() {
    return portfolio_risk;
  }
}

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

    @Override
    public Object clone() {
      Matrix other = new Matrix(rows, cols);
      for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
          other.setItem(r, c, data[r][c]);
      return other;
    }

    public int rowCount() {
      return rows;
    }

    public int colCount() {
      return cols;
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
    public static Matrix identity(int rows){
	int rCount;
	Matrix retMatrix = new Matrix(rows, rows);

	for (rCount = 0; rCount < rows; ++rCount){
	    retMatrix.setItem(rCount, rCount, 1);
	}

	return retMatrix;
    }

    //Function to create the identity matrix with a value
    public static Matrix identity(int rows, double colVal){
	int rCount;
	Matrix retMatrix = new Matrix(rows, rows);

	for (rCount = 0; rCount < rows; ++rCount){
	    retMatrix.setItem(rCount, rCount, colVal);
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
    public static Matrix add (Matrix matA, Matrix matB){
	
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

    public static double scalar_mult(Matrix A, Matrix B) {
  if (A.cols == 1 && B.cols == 1 && A.rows == B.rows) {
      double sum = 0.;
      for (int r = 0; r < A.rows; r++)
          sum += A.getItem(r, 0)*B.getItem(r, 0);

      return sum;
  }
  else throw new ArithmeticException("Incorrect matrix dimensions");
    }

    public static Matrix mult(Matrix A, double[] B_values) {
  double[][] B_values_fill = new double[B_values.length][1];
  for (int i = 0; i < B_values.length; i++)
      B_values_fill[i][0] = B_values[i];

  Matrix B = new Matrix(B_values.length, 1, B_values_fill);
  return mult(A, B);
    }

    public static Matrix mult(Matrix A, double scalar) {
      Matrix B = new Matrix(A.rows, A.cols);
      for (int r = 0; r < A.rows; r++)
        for (int c = 0; c < A.cols; c++)
          B.setItem(r, c, A.getItem(r, c)*scalar);
      return B;
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

	
    /*
     * def __rdiv__(A,x):
     *   """Computes x/A using Gauss-Jordan elimination where x is a scalar"""
     *   import copy
     *   n = A.cols
     *   if A.rows != n:
     *      raise ArithmeticError, "matrix not squared"
     *   indexes = range(n)
     *   A = copy.deepcopy(A)
     *   B = Matrix.identity(n,x)
     *   for c in indexes:
     *       for r in xrange(c+1,n):
     *           if abs(A[r,c])>abs(A[c,c]):
     *               A.swap_rows(r,c)
     *               B.swap_rows(r,c)
     *       p = 0.0 + A[c,c] # trick to make sure it is not integer
     *       for k in indexes:
     *           A[c,k] = A[c,k]/p
     *           B[c,k] = B[c,k]/p
     *       for r in range(0,c)+range(c+1,n):
     *           p = 0.0 + A[r,c] # trick to make sure it is not integer
     *           for k in indexes:
     *               A[r,k] -= A[c,k]*p
     *               B[r,k] -= B[c,k]*p
     *       # if DEBUG: print A, B
     *   return B
     */
    public static Matrix rDiv(Matrix matrix, double x) {
      int n = matrix.cols;
      if (matrix.rows != n)
        throw new RuntimeException("Matrix not square");
      Matrix A = (Matrix)matrix.clone();
      Matrix B = identity(n, x);
      for (int c = 0; c < n; c++) { 
        for (int r = c+1; r < n; r++) {
          if (Math.abs(A.getItem(r, c)) > Math.abs(A.getItem(c, c))) {
            swap_rows(A, r, c);
            swap_rows(B, r, c);
          }
        }
        double p = A.getItem(c, c);
        for (int k = 0; k < n; k++) {
          A.setItem(c, k, A.getItem(c, k)/p);
          B.setItem(c, k, B.getItem(c, k)/p);
        }
        for (int r = 0; r < n; r++) {
          if (r != c) {
            p = A.getItem(r, c);
            for (int k = 0; k < n; k++) {
              A.setItem(r, k, A.getItem(r, k)-(A.getItem(c, k)*p));
              B.setItem(r, k, B.getItem(r, k)-(B.getItem(c, k)*p));
            }
          }
        }
      }
      return B;
    }

	
	public static Matrix div(Matrix A, Matrix B){
		Matrix C = Matrix.rDiv(B, 1.);
		return mult(A, C);
		
	}
	
	
	static void swap_rows(Matrix A, int i,int j){
		double tempVal = 0;		
		for(int c=0; c<A.cols; c++){
			tempVal = A.data[i][c];			
			A.data[i][c] = A.data[j][c];
			A.data[j][c] = tempVal;
		}		
				
  	}



    //Helper function to ensure the dimensions are the same
    public static boolean checkDimensions(Matrix matA, Matrix matB){
	if ((matA.cols == matB.cols) && (matA.rows == matB.rows)){
	   return true;
	 }
	else{
	    return false;
	} 
    }

    //Subtracts the matrices, element by element, provided that the two matrices are
    //of the same dimensions 
    public static Matrix subtract(Matrix matA, Matrix matB){

    	int rCount, cCount;
    	Matrix matS = null;

    	if (checkDimensions(matA, matB)){

    	    matS = new Matrix(matA.rows, matB.rows);

    	    for (rCount = 0; rCount < matS.rows; rCount++){
    	    	for (cCount = 0; cCount < matS.cols; cCount++){
    	    		matS.setItem(rCount, cCount, matA.getItem(rCount, cCount)
    	    									- matB.getItem(rCount, cCount));
    		}
    	    }	   

    	}

    	return matS;
        }    
        
     //Adds the matrices in reverse order
    //Note:Dimensions checked by add function already
    public static Matrix rAdd(Matrix matA, Matrix matB)
    {
    	return add(matB,matA);
    	
    }
        
    //Function that reverses the order of subtraction for the two matrices by copying and negating all values of 
    //the first matrix into a new matrix, which is added to the second matrix
    //Note:Dimension is checked already by add function
    public static Matrix rSubtract(Matrix matA, Matrix matB){

    	Matrix matRS = null;

    	    matRS = new Matrix(matA.rows, matB.rows);
    	    Matrix matNA = new Matrix(matA.rows, matA.cols);
    	   
    	    for ( int rCount=0; rCount<matNA.rows; rCount++)
    	    {
    	      for ( int cCount=0; cCount<matNA.cols; cCount++ )
    	    {
    	    	  matNA.setItem(rCount,cCount, (-1) *matA. getItem(rCount,cCount));
    	    }
    	    
    	    }

    	      matRS = add(matNA, matB);

    	return matRS;
        }    
    
  //A function to negate a matrix
    public static Matrix negate(Matrix matA){
    
    	Matrix matNA = new Matrix(matA.rows, matA.cols);
    	for ( int rCount=0; rCount<matNA.rows; rCount++)
	    {
	      for ( int cCount=0; cCount<matNA.cols; cCount++ )
	      {
	    	  matNA.setItem(rCount,cCount, (-1) *matA. getItem(rCount,cCount));
	      }

	    }
    	return matNA;
    }

    //Transpose
    public static Matrix transpose (Matrix matA){

	int rCount, cCount;
	Matrix matT = new Matrix(matA.cols, matA.rows);

	for (rCount = 0; rCount < matT.rows; rCount++){
	  for (cCount = 0; cCount < matT.cols; cCount++){
	      matT.setItem(rCount, cCount, matA.getItem(cCount, rCount));
	  }
	}	   

	return matT;
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

    public static interface Function {
      public double execute(double a);
    }

    public static class Derivative implements Function {
      protected Function f;
      protected double h;

      public Derivative(Function f) {
        this.f = f;
        this.h = 0.0000001;
      }

      public Derivative(Function f, double h) {
        this.f = f;
        this.h = h;
      }

      public double execute(double a) {
        return (f.execute(a+h)-f.execute(a-h))/2/h;
      }
    }

    public static double optimize_bisection(Function f, double a, double b) {
      return optimize_bisection(f, a, b, 0.0000001);
    }
    
    public static double optimize_bisection(Function f, double a, double b, double ap) {
      return optimize_bisection(f, a, b, ap, 0.00001);
    }

    public static double optimize_bisection(Function f, double a, double b, double ap, double rp) {
      return optimize_bisection(f, a, b, ap, rp, 100);
    }
    
    /*
     * def optimize_bisection(f, a, b, ap=1e-6, rp=1e-4, ns=100):
     * Dfa, Dfb = D(f)(a), D(f)(b)
     * if Dfa == 0: return a
     * if Dfb == 0: return b
     * if Dfa*Dfb > 0:
     *   raise ArithmeticError, 'D(f)(a) and D(f)(b) must have opposite sign'
     * for k in xrange(ns):
     *   x = (a+b)/2
     *   Dfx = D(f)(x)
     *   if Dfx==0 or norm(b-a)<max(ap,norm(x)*rp): return x
     *   elif Dfx * Dfa < 0: (b,Dfb) = (x, Dfx)
     *   else: (a,Dfa) = (x, Dfx)
     * raise ArithmeticError, 'no convergence'
     */
    public static double optimize_bisection(Function f, double a, double b, double ap, double rp, int ns) {
      double Dfa = new Derivative(f).execute(a);
      double Dfb = new Derivative(f).execute(b);
      if (Dfa == 0)
        return a;
      else if (Dfb == 0)
        return b;
      else if (Dfa*Dfb > 0)
        throw new RuntimeException("First derivatives at f(a) and f(b) must have opposite sign");

      for (int k = 0; k < ns; k++) {
        double x = (a+b)/2;
        double Dfx = new Derivative(f).execute(x);
        if (Dfx == 0 || norm(b-a) < Math.max(ap, norm(x)*rp))
          return x;
        else if (Dfx * Dfa < 0) {
          b = x;
          Dfb = Dfx;
        }
        else {
          a = x;
          Dfa = Dfx;
        }
      }
      throw new RuntimeException("No convergence");
    }

    public static double optimize_secant(Function f, double x) {
      return optimize_secant(f, x, 0.0000001);
    }
    
    public static double optimize_secant(Function f, double x, double ap) {
      return optimize_secant(f, x, ap, 0.00001);
    }

    public static double optimize_secant(Function f, double x, double ap, double rp) {
      return optimize_secant(f, x, ap, rp, 100);
    }

    /*
     * def optimize_secant(f, x, ap=1e-6, rp=1e-4, ns=100):
     *   x = float(x) # make sure it is not int
     *   (fx, Dfx, DDfx) = (f(x), D(f)(x), DD(f)(x))
     *   for k in xrange(ns):
     *     if Dfx==0: return x
     *     if norm(DDfx) < ap:
     *       raise ArithmeticError, 'unstable solution'
     *     (x_old, Dfx_old, x) = (x, Dfx, x-Dfx/DDfx)
     *     if norm(x-x_old)<max(ap,norm(x)*rp): return x
     *     fx = f(x)
     *     Dfx = D(f)(x)
     *     DDfx = (Dfx - Dfx_old)/(x-x_old)
     *   raise ArithmeticError, 'no convergence'
     */
    public static double optimize_secant(Function f, double x, double ap, double rp, int ns) {
      double fx = f.execute(x);
      double Dfx = new Derivative(f).execute(x);
      double DDfx = new Derivative(new Derivative(f)).execute(x);
      for (int k = 0; k < ns; k++) {
        if (Dfx == 0)
          return x;
        else if (norm(DDfx) < ap)
          throw new RuntimeException("unstable solution");

        double x_old = x;
        double Dfx_old = Dfx;
        x = x-Dfx/DDfx;

        if (norm(x-x_old) < Math.max(ap, norm(x)*rp))
          return x;
        fx = f.execute(x);
        Dfx = new Derivative(f).execute(x);
        DDfx = (Dfx - Dfx_old)/(x-x_old);
      }
      throw new RuntimeException("no convergence");
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
	
	testMatrix = newMatrix.diagonal(vals);
	newMatrix = newMatrix.identity(4,1);
	
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

	Matrix a = Matrix.identity(2,1);

  double[][] cov_data = {{0.04, 0.006,0.02},
                         {0.006,0.09, 0.06},
                         {0.02, 0.06, 0.16}};

  Matrix cov = new Matrix(3, 3, cov_data);
  System.out.println("cov:");
  cov.printMatrix();
  System.out.println("rDiv:");
  Matrix.rDiv(cov, 1.).printMatrix();

  double[][] mu_data = {{0.10}, {0.12}, {0.15}};
  double r_free = 0.05;

  Matrix mu = new Matrix(3, 1, mu_data);
  Markovitz mark = new Markovitz(mu, cov, r_free);
  System.out.println("markovitz portfolio:");
  double[] portfolio = mark.getPortfolio();
  System.out.print("[");
  for (int i = 0; i < portfolio.length-1; i++)
    System.out.print(portfolio[i]+", ");
  System.out.println(portfolio[portfolio.length-1]+"]");

  Function f = new Function() {
    public double execute(double a) {
      return (a-2.)*(a-5.);
    }
  };
  System.out.print("optimize_bisection: ");
  double point = optimize_bisection(f, 2., 5.);
  System.out.println(point);
  System.out.print("optimize_secant: ");
  point = optimize_secant(f, 3.);
  System.out.println(point);
    }
}

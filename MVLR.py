# -*- -*-
"""
Author: Shaon Majumder
This is multivariate linear regression implementation from scratch
"""
import matplotlib.pyplot as plt
import numpy
## Statistical Function
def list_multiplication(dm1,dm2):
	if length(dm1) == length(dm2):
		return [b*c for b,c in zip(dm1,dm2)]
	elif length(dm1) == 1 or length(dm2) == 1:
		if length(dm1) == 1:
			r = dm1[0]
			c = dm2
		elif length(dm2) == 1:
			r = dm2[0]
			c = dm1
		return [i*r for i in c]
	else:
		print("shape is not same for list multiplication")
		raise ValueError

def ArithmeticMean(li=[],lower=[],upper=[],frequency=[]):
	if li == []:
		for i in range(length(lower)):
			midpoint =(lower[i]+upper[i])/2
			li = appends(li,midpoint)
	else:
		del lower
		del upper
		del frequency
		frequency = []
	
	if frequency == []:
		for i in range(length(li)):
			frequency = appends(frequency,1)
	sumn = 0
	iteration = 0
	for i in range(length(li)):
		sumn += (li[i]*frequency[i])
		iteration  += frequency[i]
	return float(sumn)/iteration

def sample_standard_deviation(li):
	return (sum([(i - ArithmeticMean(li))**2 for i in li])/(length(li)-1))**(1/float(2))
## Statistical Function ends

## String Function
def length(li):
	iteration = 0
	for c in li:
		iteration += 1
	return iteration

def split(string,spliting_char = ','):
	#Spliting Functions by comma, or character. Default spliting character is comma.
	word = ""
	li = []
	iteration = 0

	for c in string:
		if c != spliting_char:
			word += c
			#if c == '':
			#	c = null_string
		else:			
			li = appends(li,word)
			word = ""
		iteration += 1
	#if word != "":
	li = appends(li, word)
	return li

def strip(string,strip_chars = [" "]):
	iters = True
	iteration = 0
	result = ""
	rstat = False
	for c in string:
		
		for d in strip_chars:
			if c != d:
				rstat = True
			else:
				rstat = False
				break

		if rstat == True:
			result += c
			rstat = False
		

		iteration += 1
	return result

def appends(lst, obj, index = -2):
	if index == -2:# -2 for accepting 0 value to be passed as it represents begining of the array
		index = length(lst)

	if type(obj) is list:
		return lst[:index] + obj + lst[index:]
	else:
		return lst[:index] + [obj] + lst[index:]

def conv_type(obj,type_var):
	dts = ["int","float","str"]
	st = 0
	for c in dts:
		if type_var == c:
			st = 1
			break

	if st != 1:
		raise Exception('No avaiable conversion type passed')

	if type(obj) is list:
		#print("list")
		pass
	elif type(obj) is str:
		print("string")
	elif type(obj) is int:
		print("Integer")
	elif type(obj) is float:
		print("Float")
	else:
		print("else %s" % type(obj))

	lists = []
	callables = eval(type_var)
	for c in obj:
		try:
			lists = appends(lists,callables(c))
		except ValueError:
			lists = appends(lists,c)

	return lists
## String Function ends


## Data Manipulation
def getIndex(li,val):
	#indexes = [i for i,x in enumerate(li) if x == val]
	#indexes[-1]
	iteration = 0
	for c in li:
		if(val == li[iteration]):
			return iteration
		iteration += 1
	return False
def reference_reverse_normalize(ypure,y_pred):
	y_PRED = []
	ystd = sample_standard_deviation(ypure)
	ymean = ArithmeticMean(ypure) 
	for c in range(length(y_pred)):
		y_reverse = y_pred[c]* ystd + ymean
		y_PRED.append(y_reverse)
	return y_PRED


def df_size(df):#obs
	row = length(df[0])
	column = length(df)
	return row, column


def transpose(dm):
	n,m = df_size(dm)
	dm_n = create_dataframe(m,n)
	for c in range(m):
		it = 0
		for b in dm[c]:
			#print("dm_n[%s][%s] = %s" % (it,c,b))
			dm_n[it][c] = b
			it += 1
	return dm_n

def create_dataframe(m,n):#obs
	'''
	df = []
	
		
	df_r = []
	for i in range(m):
		df_r = appends(df_r,0)

	for c in range(n):
		df = appends(df,[df_r])
	'''	
	df=[]
	df = [[None]*m for _ in range(n)]
	return df



class DataFrame(object):
	'''This is Onnorokom General Library'''    
	def __init__(self, columns=[], dataframe = ['0']):
		#sequence checked
		self.dataframe = dataframe
		self.shape = self.framesize
		#self.T = self.trans
		self.columns = columns
		
		if self.dataframe != ['0']:
			self.shape = self.framesize
			self.T = self.trans
			if columns == []:
				self.columns = self.erase_col_names()
			

	def __del__(self):
		classname = self.__class__.__name__
		#print ('%s class destroyed' % (classname))

	def __str__(self):
		#return str(self.columns) + str(self.dataframe)
		#print(self.dataframe) #List representation
		strs = "Dataframe Representation\n"
		
		def str_sp(strs,space=10):
			
			for c in range(space - length(strs)):
				strs += " "
			return strs

		for c in self.columns:
			strs += str_sp(c)
		strs += "\n"
		for c in range( length(self.dataframe[0]) ):
			for d in self.dataframe:
				strs += str_sp( str(d[c]) )
				
			strs += "\n"
		return strs
	def __getitem__(self,name):
		if type(name) == int:
			
			return self.dataframe[name]
		elif type(name) == str:
			if name.isdigit() == True:
				return self[int(name)]
			
			return self.dataframe[getIndex(self.columns,name)]
		elif isinstance(name, slice):
			return self.dataframe[name]
	def __iter__(self):
	    return iter(self.columns)
        
	def normalize(self,change_self=False):
		#we need to normalize the features using mean normalization
		df = []
		for c in self.dataframe:
			mean= ArithmeticMean(c) 
			std = sample_standard_deviation(c)
			_ =[ (a - mean)/std for a in c]
			df = appends(df,[_])
		if change_self == True:
			self.dataframe = df
		return df
	def conv_type(self,var_type,change_self=False):
		callables = eval(var_type)
		df = []
		col = []
		for c in self.dataframe:
			for i in c:

				col = appends(col,callables(i))
			df = appends(df,[col])
			col = []

		if change_self == True:
			self.dataframe = df
		return df
    	
	def ix(self):
		pass
	def iloc(self):
		pass
	def row(self,rowindex):
		row = []
		for c in self.columns:
			row.append(self[c][rowindex])
		return row
	def new(self,m,n,elm=''):
		if elm == '':
			elm = None
		df=[]
		df = [[elm]*m for _ in range(n)]
		#if change_self == True:
		return self.set_object(df)
	def concat(self,dm1,dm2,axis=0):
		#axis
		#[0 - row merge]
		#[1 - column merge]
		dm = []
		m,n = dm1.framesize
		x,y = dm2.framesize
		b = dm1.tolist
		d = dm2.tolist
		if axis == 0:
			if n != y:
				print('ValueError: all the input array dimensions except for the concatenation axis must match exactly')
				raise ValueError
			for c,a in zip(b,d):
				dm = appends(dm,[c+a])
		elif axis == 1:
			if m != x:
				print('ValueError: all the input array dimensions except for the concatenation axis must match exactly')
				raise ValueError
			
			for c in b:###
				dm = appends(dm,[c])
			
			for c in d:###
				dm = appends(dm,[c])
		return self.set_object(dm)
	def transpose(self,change_self=False):
		selfs = self.__class__()
		dm = self.dataframe
		n,m = self.size()
		dm_n = create_dataframe(m,n)
		for c in range(m):
			it = 0
			for b in dm[c]:
				#print("dm_n[%s][%s] = %s" % (it,c,b))
				dm_n[it][c] = b
				it += 1
		if change_self == True:
			self.dataframe = dm_n
			self.erase_col_names()
			#self.T = dm_n #previous_code 5-4-18
			#self.T = self.trans #previous_code 5-4-18
		
		#sequence for returing self after dataframe calculation
		selfs.dataframe	= dm_n
		selfs.shape = selfs.framesize
		selfs.columns = selfs.erase_col_names()
		return selfs
		#sequence for returing self after dataframe calculation
	def erase_col_names(self):
		self.columns = [str(c) for c in range(length(self.dataframe))]
		return self.columns
	def columns(self):
		return [ c for c in self.columns]
	def size(self,ob=[]):
		if ob == []:
			return length(self.dataframe[0]),length(self.dataframe)
		else:
			return length(ob[0]),length(ob)
	def read_csv(self,input_file,columns=[]):
		with open(input_file,'r') as file:#Taking file input as stream, handly for iterators and reading stream object saves memory losses
			data = file.readlines()#reading line by line

		first_line = split( strip(data[0],[' ','\n']) ,",")


		header = [c for c in range(0,length(first_line))]			
		
		if first_line[0].isdigit() == False:
			self.columns = first_line
			del data[0]
		else:
			self.columns = conv_type(header,"str")

		df = [[] for d in header]

		for c in data:
			line = split( strip(c,[' ','\n']) ,',')
			for d in header:
				df[d] = appends(df[d],line[d])

		if columns == []:
			columns = self.columns
		else:
			self.columns = columns
		
		#sequence for returing self after dataframe assigning
		self.dataframe = df
		self.shape = self.framesize
		self.T = self.trans
		#sequence for returing self after dataframe assigning

		return self
	def __sub__(self,dm2):
		dm1 = self
		selfs = self.__class__()
		m,n = dm1.shape
		x,y = dm2.shape
		dm=[]
		a = dm1.tolist
		b = dm2.tolist
		
		if m == x and n == y:
			j = 0
			for c in range(n):
				i = 0
				col = []
				for r in range(m):
					si = a[j][i] - b[j][i]
					i+=1
					col.append(si)
				j+=1	
				dm.append(col)
			return self.set_object(dm)
		else:
			print("Matrice Shape is not same for substract",dm1.shape,dm2.shape)
			raise ValueError
	def __add__(self,dm2):
		dm1 = self
		selfs = self.__class__()
		m,n = dm1.shape
		x,y = dm2.shape
		dm=[]
		a = dm1.tolist
		b = dm2.tolist
		
		if m == x and n == y:
			j = 0
			for c in range(n):
				i = 0
				col = []
				for r in range(m):
					si = a[j][i] + b[j][i]
					i+=1
					col.append(si)
				j+=1	
				dm.append(col)
			return self.set_object(dm)
		else:
			print("Matrice Shape is not same for substract",dm1.shape,dm2.shape)
			raise ValueError
	def __gt__(self, other):
		pass
	def __lt__(self, other):
		pass
	def __ge__(self, other):
		pass
	def __le__(self, other):
		pass
	def two2oneD(self):
		if self.shape[0] == 1:
			return self.T[0]
		elif self.shape[1] == 1:
			return self[0]
		else:
			print("Column/row != 1, can not converted to 1D list")

	def dot(self,dm1,dm2):
		a,b=dm1.shape
		m,n=dm2.shape
		if(b==m):
			dm_n = []
			it = 0
			for c in dm2.tolist:
				col = []
				for i in (dm1.T).tolist:
					col = appends(  col,sum( list_multiplication(i,c) )  )
				dm_n = appends(dm_n,[col])
				it+=1
			return self.set_object(dm_n)
		else:
			print("Shape is not same for matrice multiplication -> ",dm1.shape,dm2.shape)
			raise ValueError
	def __mul__(self,dm2):
		dm1 = self
		#used inverse dataframe to convert numpy array representation
		#then used numpy broadcasting
		c1 = (dm1.shape[0]==dm2.shape[0]) and (dm1.shape[1] == 1 or dm2.shape[1] == 1)
		c2 = (dm1.shape[1]==dm2.shape[1]) and (dm1.shape[0] == 1 or dm2.shape[0] == 1)
		if dm1.shape == dm2.shape or c1:
			dm = []
			dm1 = (dm1.T).tolist
			dm2 = (dm2.T).tolist
			for r1,r2 in zip(dm1,dm2):
				dm.append(list_multiplication(r1,r2))
			#they are equal, or
			#one of them is 1
			return self.set_object(transpose(dm))
		elif c2:
			dm = []
			dm1 = (dm1.T).tolist
			dm2 = (dm2.T).tolist
			for r1 in dm1:
				dm = [list_multiplication(r1,r2) for r2 in dm2]
			#they are equal, or
			#one of them is 1
			return self.set_object(transpose(dm))
		else:
			print("cross is not allowed")
			raise ValueError
	def sum(self,axis=0):
		if axis == 0:
			dm = [[sum(c)] for c in self.tolist]
		elif axis == 1:
			dm = [[sum(self.row(d)) for d in range(self.shape[0])]]
		return self.set_object(dm)
		
	def sum_np(self,axis=0):
		return self.sum(axis).two2oneD()
	def __float__(self):
		if self.shape == (1,1):
			return self[0][0]
	def __pow__(self,power):
		dm = [ [r**power for r in c] for c in self.tolist]
		return self.set_object(dm)
	def dftolist(self):
		return self.dataframe
	def dataA(self,dataframe):
		self.dataframe = dataframe
		return self
	def set_object(self,dm):
		selfs = self.__class__()
		#sequence for returing self after dataframe calculation
		selfs.dataframe	= dm
		selfs.shape = selfs.framesize
		selfs.T = selfs.trans
		selfs.columns = selfs.erase_col_names()
		return selfs
		#sequence for returing self after dataframe calculation

	framesize = property(size)
	tolist = property(dftolist)
	trans = property(transpose)
	prop_var =property(set_object)


## ML Function
def minResidual(pure , pred):
	"""returns minimum error distance or residual"""
	E = []
	for c in range(length(pure)):
		absolute_distance = abs(pure[c] - pred[c])
		E.append(absolute_distance)

	return min(E)
def meanResidual(pure , pred):
	"""returns average error distance or residual"""
	E = []
	for c in range(length(pure)):
		absolute_distance = abs(pure[c] - pred[c])
		E.append(absolute_distance)
	import numpy as np
	return np.mean(E)

def maxResidual(pure , pred):
	"""returns maximum error distance or residual"""
	E = []
	for c in range(length(pure)):
		absolute_distance = abs(pure[c] - pred[c])
		E.append(absolute_distance)

	return max(E)

def give_time_series(x,y):
	"""Rearrange X,Y value pairs or points according to X's order"""
	xall = []
	yall = []
	for x1,y1 in sorted(zip(x,y)):
		xall.append(x1)
		yall.append(y1)
	return (xall,yall)

def plot_error_distance(x,y_pred,y_actual):
	"""Plot error distance or residual"""
	for [a,b,c] in [ [x,y_r,y_p] for x,y_r,y_p in zip(x,y_pred,y_actual) ]:
		plt.plot([a,a],[b,c],color='y',label='residual')

def MLVR(XDATA,YDATA,xreference=0,residual=1,xlabel='',ylabel='',title='',alpha = 0.01,iters = 1000,plot=1):	
	"""Does Multivariant Linear Regression
	properties:
		XDATA = The Feature Dataframe
		YDATA = The Target Dataframe
		xreference = 1/0 -> The column index in XDATA for ploting graph
		xlabel = Label for X in Graph
		ylabel = Label for Y in Graph
		title = title for graph]
		alpha = Learning rate for model
		iters = the number of iteration to train the model
	"""
	XDATA.conv_type('float',change_self=True)
	xpure = XDATA[xreference]
	XDATA.normalize(change_self=True)

	YDATA.conv_type('float',change_self=True)
	ypure = YDATA.tolist[0]
	YDATA.normalize(change_self=True)

	X=XDATA
	y=YDATA

	df =DataFrame()
	ones = df.new(X.shape[0],1,elm=1.)
	X = df.concat(ones,X,axis=1)
	
	theta = DataFrame().new(1,length(X.columns),elm=0.)
	
	def computeCost(X,y,theta):
		dot_product = DataFrame().dot(X,theta.T)	
		return float(    (  (dot_product - y)**2  ).sum(axis=0)    )/(2 * X.shape[0])
	
	def gradientDescent(X,y,theta,iters,alpha):
		#cost = np.zeros(iters)
		cost = []
		for i in range(iters):			
			dot_product = DataFrame().dot(X,theta.T)
			derivative = DataFrame(dataframe = [[(alpha/X.shape[0])]])  *  ( X*(dot_product - y) ).sum(axis = 0 ) 
			theta = theta - derivative			
			cost.append( computeCost(X, y, theta) ) #cost[i] = computeCost(X, y, theta)
		return theta,cost

	def print_equation(g):
		stra = "Estimated equation, y = %s"%g[0]
		g0 = g[0]
		del g[0]
		for c in range(length(g)):
			stra += " + %s*x%s"%(g[c],c+1)
		print(stra)

	def predict_li(XDATA,g):
		g0 = g[0]
		del g[0]
		y_pred = []			
		for row in range(XDATA.shape[0]):
			suma = 0
			suma += sum(list_multiplication( g , XDATA.row(row) ) )
			yres = g0 + suma
			y_pred.append(yres)	
		return y_pred

	g,cost = gradientDescent(X,y,theta,iters,alpha)
	finalCost = computeCost(X,y,g)
	#g = g.T
	g = g.two2oneD()
	print("Thetas = %s"%g) #print("cost = ",cost)
	print("finalCost = %s" % finalCost)
	gN = g[:]
	print_equation(gN)

	gN = g[:]	
	y_pred = predict_li(XDATA,gN)	
	
	y_PRED = reference_reverse_normalize(ypure,y_pred)
	emin,emean,emax = minResidual(ypure , y_PRED),meanResidual(ypure , y_PRED),maxResidual(ypure , y_PRED)
	
	print("Min,Mean,Max residual = %s, %s, %s"%( emin,emean,emax ) )
	print("Residual Min - Max Range = %s"%(emax-emin))
	print("Residual range percentage = %s" %((emax-emin)/(max(ypure) - min(ypure))) )
	
	print("Residual mean percentage = %s" %(emean/ArithmeticMean(ypure)) )




	#-- If finalcost is lowest mean Residual or mean Error distance also will be lowest


	#y_pred = [g[0] + g[1]*my_data[0][c] + g[2]*my_data[1][c] for c in range(my_data.shape[0])]
	y_actual = YDATA.tolist[0]
	x = XDATA[xreference]

	if plot == 1:
		fig, ax = plt.subplots()  
		ax.plot(numpy.arange(iters), cost, 'r')  
		ax.set_xlabel('Iterations')  
		ax.set_ylabel('Cost')  
		ax.set_title('Error vs. Training Epoch')  
		plt.show()

		x_a, y_a = give_time_series(xpure,y_PRED)#give_time_series(x,y_pred)
		plt.plot(x_a,y_a,color='r',marker='.',label='Prediction')

		x_a, y_a = give_time_series(xpure,ypure)#give_time_series(x,y_actual)
		plt.plot(x_a,y_a,color='g',marker='.',label='Real')
		
		if residual == 1:
			plot_error_distance(xpure,y_PRED,ypure)

		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.title(title)
		plt.legend()
		plt.show()
	else:
		print('plot off')
		
	return finalCost


def main():	
	my_data = DataFrame().read_csv('sample_inputs/home.txt',columns=["size","bedroom","price"])
	
	XDATA = DataFrame(dataframe= my_data[0:2],columns=['size','bedroom'])
	YDATA = DataFrame(dataframe= [my_data[2]])
	
	MLVR(XDATA,YDATA,xreference=0,residual=1,xlabel='serial,length',ylabel='views',title='Youtube View MLVR',alpha = 0.01,iters = 1000)

if __name__ == "__main__":
    main()

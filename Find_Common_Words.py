def Find(x,y):
	return(list(set(x.split()).intersection(set(y.split()))))

def Find_Common_Words(a,b):
	return(list(map(Find,a,b)))


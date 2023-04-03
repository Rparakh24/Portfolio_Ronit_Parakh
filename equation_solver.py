x1 = int(input("Enter the coffecient of x1 in equation:- "))
y1 = int(input("Enter the coffecient of y1 in equation:- "))
c1 = int(input("Enter the coffecient of c1 in equation:- "))
x2 = int(input("Enter the coffecient of x2 in equation:- "))
y2 = int(input("Enter the coffecient of y2 in equation:- "))
c2 = int(input("Enter the coffecient of c2 in equation:- "))
if(x1==x2):
    a = y1-y2
    b=c1-c2
    y=b/a
else:
    x1_new=x1*x2
    x2_new=x2*x1
    y1_new=y1*x2
    y2_new=y2*x1
    c1_new=c1*x2
    c2_new=c2*x1
    if(y1_new>=y2_new):
      a=y1_new-y2_new
      b=c1_new-c2_new
    else:
      a=y2_new-y1_new
      b=c2_new-c1_new
    y=b/a
            

x=(c1_new-(y*y1_new))/x1_new
print(x)
print(y)

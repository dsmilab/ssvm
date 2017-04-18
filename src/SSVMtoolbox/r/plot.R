#curve
plot(test[,1], type="l", )

#pie
sales=c(0.12,0.3,0.26,0.16,0.04,0.12)
snames=c("電腦", "廚房家電", "女性服飾", "客廳家具", "其他", "男性服飾")
pie(sales, label=snames)

#boxplot
boxplot(iris[,1], xlab="Sepal.Length", main="boxplot() test", col="gray")

#histogram
x=rnorm(100)
b=c(-3,-2,-1,0,1,2,3)
hist(x, breaks=b)
hist(x, nclass=8, col="Green")


#3D
x=seq(-3, 3, 0.1)
y=x
f=function(x,y){1/(2*pi)*exp(-0.5*(x^2+y^2))}
z=outer(x,y,f)
par(mfcol=c(2,2))
contour(x,y,z)
image(x,y,z,)
persp(x,y,z)
persp(x,y,z,theta=30, phi=30, box=F, main="theta=30, phi=30")



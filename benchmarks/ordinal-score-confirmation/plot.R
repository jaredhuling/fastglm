root <- "benchmarks/ordinal-score-confirmation"
z <- read.csv(file.path(root,"rates.csv")); z <- z[z$null & z$level==.05,]
labels <- c(balanced4="Balanced, four categories",endpoint4="Rare endpoint",
    narrow_middle4="Narrow interior category",correlated6="Correlated predictors, six categories",
    sparse_binary4="Uncommon binary predictor",leverage4="High leverage",grouped4="Grouped frequencies")
pdf(file.path(root,"null-rejection.pdf"),width=10,height=12,family="Helvetica",useDingbats=FALSE,bg="white")
par(mfrow=c(4,2),mar=c(4,4,3,1),oma=c(1,0,2,0))
colors <- c(joint="#17628D",x1="#B55D15")
for(scenario in names(labels)) {
    plot(NA,xlim=c(.7,3.3),ylim=c(0,10),xaxt="n",xlab="Individual frequency",
         ylab="Rejection rate (%)",main=labels[scenario])
    axis(1,1:3,c(100,400,1600)); abline(h=5,col="gray50",lty=2)
    for(test in names(colors)) {
        g <- z[z$scenario==scenario & z$test==test,]; g <- g[order(g$n),]
        x <- 1:3+if(test=="joint") -.05 else .05
        segments(x,100*g$lower,x,100*g$upper,col=colors[test])
        lines(x,100*g$rate,col=colors[test],type="b",pch=if(test=="joint") 16 else 17)
    }
}
plot.new()
legend("top",legend=c("Joint test","First predictor","Nominal 5%"),col=c(colors,"gray50"),
       pch=c(16,17,NA),lty=c(1,1,2),bty="n",cex=1.1)
text(.5,.45,paste("Rates condition on available tests.","Vertical lines: 95% Wilson Monte Carlo intervals.",
    "1000 datasets per scenario and sample size.","Availability and missing-decision bounds: rates.csv.",
    "Post-confirmation precision checks are reported separately.",sep="\n"),cex=.9)
mtext("Original confirmation: expected-information score tests",outer=TRUE,cex=1.2)
invisible(dev.off())

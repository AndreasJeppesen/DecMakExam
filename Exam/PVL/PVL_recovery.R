# PVL recovery
install.packages("pacman")
pacman::p_load(extraDistr, R2jags, parallel, ggpubr, ggplot2)

set.seed(1983)

### NB! Don't forget to set your working directory
setwd('/work/PVL')

# defining a function for calculating the maximum of the posterior density (not exactly the same as the mode)
MPD <- function(x) {
  density(x)$x[which(density(x)$y==max(density(x)$y))]
}

#------ create task environment -------------------
# NB! mod(ntrials, nstruct) (aka. ntrials %% nstruct) must be 0
ntrials <- 100 # total number of trials in our payoff structure
nstruct <- 10 # size of our subdivisions for pseudorandomization
freq <- 0.5 # probability of our frequent losses (we have losses half of the time)
infreq <- 0.1 # probability of our infrequent losses (we have losses 1/10th of the time)
bad_r <- 100 # "bad" winnings
bad_freq_l <- -250 # "bad" frequent loss
bad_infreq_l <- -1250 # "bad" infrequent loss
good_r <- 50 # "good" winnings
good_freq_l <- -50 # "good" frequent loss
good_infreq_l <- -250 # "good" infrequent loss

# Bad frequent
A_R <- rep(bad_r, nstruct) # we win on every trials
A_L <- c(rep(bad_freq_l, nstruct*freq),rep(0,nstruct*(1-freq))) # we have losses half of the time

# Bad infrequent
B_R <- rep(bad_r, nstruct)
B_L <- c(rep(bad_infreq_l, nstruct*infreq),rep(0,nstruct*(1-infreq))) # we have losses 1/10th of the time

# Good frequent
C_R <- rep(good_r, nstruct)
C_L <- c(rep(good_freq_l, nstruct*freq),rep(0,nstruct*(1-freq)))

# Good infrequent
D_R <- rep(good_r, nstruct)
D_L <- c(rep(good_infreq_l, nstruct*infreq),rep(0,nstruct*(1-infreq)))

# create the pseudorandomized full payoff structure
A <- array(NA,ntrials) # setting up and empty array to be filled
B <- array(NA,ntrials)
C <- array(NA,ntrials)
D <- array(NA,ntrials)
for (i in 1:(ntrials/nstruct)) {
  A[(1+(i-1)*nstruct):(i*nstruct)] <- (A_R + sample(A_L)) # randomly shuffling the loss-array for every ten trials (and adding those losses to the winnings)
  B[(1+(i-1)*nstruct):(i*nstruct)] <- (B_R + sample(B_L))
  C[(1+(i-1)*nstruct):(i*nstruct)] <- (C_R + sample(C_L))
  D[(1+(i-1)*nstruct):(i*nstruct)] <- (D_R + sample(D_L))
}

# "less generic" code
# A <- c()
# B <- c()
# C <- c()
# D <- c()
# for (i in 1:10) {
#   A <- (append(A,A_R + sample(A_L)))
#   B <- (append(B,B_R + sample(B_L)))
#   C <- (append(C,C_R + sample(C_L)))
#   D <- (append(D,D_R + sample(D_L)))
# }

payoff <- cbind(A,B,C,D)/100 # combining all four decks as columns with each 100 trials - dividing our payoffs by 100 to make the numbers a bit easier to work with

# let's look at the payoff
colSums(payoff) # the two bad decks should sum to -25 (i.e. -2500), and the two good ones to 25 (i.e. 2500)
#----------------------------------------------------


#-------test PVL delta function and jags script ---------

#---set params

w <- 2 # weighting parameter (aka. loss aversion)
A <- .5 # shape parameter (aka. risk aversion)
theta <- 3 # inverse heat parameter (aka. choice consitency)
a <- .1 # learning rate parameter (aka. prediction error weighting)

#ntrials <- 100 # we have already specified this earlier (see line 15)

source("PVL.R")
PVL_sims <- PVL(payoff,ntrials,w,A,a,theta)

par(mfrow=c(2,2))
plot(PVL_sims$Ev[,1], ylim=c(-1,1))
plot(PVL_sims$Ev[,2], ylim=c(-1,1))
plot(PVL_sims$Ev[,3], ylim=c(-1,1))
plot(PVL_sims$Ev[,4], ylim=c(-1,1))
title(paste("Traces of expeceted value (Ev) for all four decks over", ntrials, "trials"), line = -1, outer = TRUE)

x <- PVL_sims$x
X <- PVL_sims$X

# set up jags and run jags model
data <- list("x","X","ntrials")  # Remember to add "win" and "loss" for the PVL2D model
params<-c("w","A","theta","a")
temp_samples <- jags.parallel(data, inits=NULL, params,
                model.file ="PVL.txt",
                n.chains=3, n.iter=5000, n.burnin=1000, n.thin=1,
                n.cluster=3)

recov_w <- temp_samples$BUGSoutput$sims.list$w
recov_A <- temp_samples$BUGSoutput$sims.list$A
recov_a <- temp_samples$BUGSoutput$sims.list$a
recov_theta <- temp_samples$BUGSoutput$sims.list$theta

par(mfrow=c(4,1))
plot(density(recov_w))
plot(density(recov_A))
plot(density(recov_a))
plot(density(recov_theta))
title(paste("Density plots (for recovered w, A, a & theta) with ntrials =", ntrials), line = -1, outer = TRUE)


###--------------Run full parameter recovery -------------
niterations <- 100 # fewer because it takes too long
true_w <- array(0,c(niterations))
true_A <- array(0,c(niterations))
true_a <- array(0,c(niterations))
true_theta <- array(0,c(niterations))

infer_w <- array(0,c(niterations))
infer_A <- array(0,c(niterations))
infer_a <- array(0,c(niterations))
infer_theta <- array(0,c(niterations))

# checking the runtime on our parameter recovery
start_time = Sys.time()

for (i in 1:niterations) {
  
  # let's see how robust the model is. Does it recover all sorts of values?
  w <- runif(1,.5,2.5)
  A <- runif(1,0,1) 
  a <- runif(1,0,1)
  theta <- runif(1,0,5)
  
  PVL_sims <- PVL(payoff,ntrials,w,A,a,theta)
  
  x <- PVL_sims$x
  X <- PVL_sims$X
  
  # set up jags and run jags model
  data <- list("x","X","ntrials") 
  params<-c("w","A","a","theta")
  samples <- jags.parallel(data, inits=NULL, params,
                  model.file ="PVL.txt",
                  n.chains=3, n.iter=5000, n.burnin=1000, n.thin=1,
                  n.cluster=3)
  
  
  true_w[i] <- w
  true_A[i] <- A
  true_theta[i] <- theta
  true_a[i] <- a
  
  # find maximum a posteriori
  Q <- samples$BUGSoutput$sims.list$w
  infer_w[i] <- MPD(Q)
  # infer_w[i] <-density(Q)$x[which(density(Q)$y==max(density(Q)$y))]
  
  Q <- samples$BUGSoutput$sims.list$A
  infer_A[i] <- MPD(Q)
  # infer_A_A[i] <-density(Q)$x[which(density(Q)$y==max(density(Q)$y))]
  
  Q <- samples$BUGSoutput$sims.list$a
  infer_a[i] <- MPD(Q)
  # infer_a[i] <-density(Q)$x[which(density(Q)$y==max(density(Q)$y))]
  
  Q <- samples$BUGSoutput$sims.list$theta
  infer_theta[i] <- MPD(Q)
  # infer_theta[i] <-density(Q)$x[which(density(Q)$y==max(density(Q)$y))]
  
  print(i)
  
}

end_time = Sys.time()
print("Runtime for RW model: ")
end_time - start_time

# let's look at some scatter plots
par(mfrow=c(2,2))
plot(true_w,infer_w)
plot(true_A,infer_A, ylim=c(0,1))
plot(true_a,infer_a)
plot(true_theta,infer_theta)


# plotting code courtesy of Lasse
source('recov_plot.R')
pl1 <- recov_plot(true_w, infer_w, c("true w", "infer w"), 'smoothed linear fit')
pl2 <- recov_plot(true_A, infer_A, c("true A", "infer A"), 'smoothed linear fit')
pl3 <- recov_plot(true_a, infer_a, c("true a", "infer a"), 'smoothed linear fit')
pl4 <- recov_plot(true_theta, infer_theta, c("true theta", "infer theta"), 'smoothed linear fit')
ggarrange(pl1, pl2, pl3, pl4)

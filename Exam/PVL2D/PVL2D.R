PVL2D <- function(payoff,ntrials,w,A,a,theta) {  ### !!!! payoff should have a different structure - include win and loss !!
  ## Kunne være et 3D array med dimensionerne ntrials x ndecks x 2, hvor første matrix er win og anden er loss.
  ## Dvs find trial 1, deck 2, win ved payoff[1,2,1]
  
  # arrays to populate for simulation
  x <- array(NA,c(ntrials)) # choice
  X <- array(NA,c(ntrials)) # net outcome
  win <- array(NA,c(ntrials)) # wins
  loss <- array(NA,c(ntrials)) # losses
  u <- array(NA,c(ntrials)) # utility
  v <- array(NA,c(ntrials)) # valence
  uv <- array(NA,c(ntrials)) # avg of u and v
  Ev <- array(NA,c(ntrials,4)) # Expected utility
  Ev_update <- array(NA,c(ntrials,4))
  exp_p <- array(NA,c(ntrials,4)) # softmax intestines
  p <- array(NA,c(ntrials,4)) # probabilities
  
  
  
  x[1] <- rcat(1,c(.25,.25,.25,.25)) # assigning a "flat" probability structure to the first choice (i.e. random choice between the four decks)
  
  # win and loss on first trial
  win[1] <- payoff[1, x[1], 1]
  loss[1] <- payoff[1, x[1], 2]
  
  # net outcome on first trial
  X[1] <- payoff[1, x[1], 1] + payoff[1, x[1], 2]
  
  Ev[1,] <- rep(0,4) # assigning zero as the expected value for all four decks at the first "random" choice 
  
  for (t in 2:ntrials) {
    
    u[t] <- ifelse(X[t-1]<0,-(w / (1-w))*abs(X[t-1])^A,X[t-1]^A) # utility with the new factor for loss aversion
    
    v[t] <- (1-w)*win[t-1] - w*loss[t-1]  # valence with weight on win and loss
    
    uv[t] <- mean(u[t], v[t])  # mean of utility and valence
    
    for (d in 1:4) {
      
      Ev_update[t,d] <- Ev[t-1,d] + (a * (uv[t] - Ev[t-1,d])) # Delta learning rule
      
      Ev[t,d] <- ifelse(x[t-1]==d,Ev_update[t,d],Ev[t-1,d])  # only update Ev of the chosen deck
      
      exp_p[t,d] <- exp(theta*Ev[t,d]) # for the input of softmax
      
    }
    
    for (d in 1:4) {
      p[t,d] <- exp_p[t,d]/sum(exp_p[t,]) # softmax
    }
    
    x[t] <- rcat(1,p[t,]) # make choice of deck
    
    # net outcome
    X[t] <- payoff[t,x[t],1] + payoff[t,x[t],2]
    
    # win and loss
    win[t] <- payoff[t, x[t], 1]
    loss[t] <- payoff[t, x[t], 2]
    
  }
  
  result <- list(x=x,
                 X=X,
                 win = win,
                 loss = loss,
                 Ev=Ev)
  
  return(result)
  
  
  #turn back on when building
  #par(mfrow=c(2,2))
  #plot(Ev[,1])
  #plot(Ev[,2])
  #plot(Ev[,3])
  #plot(Ev[,4])
  #plot(x)
  
}

rm(list=ls())
# Import relevant packages ------------------------------------------------
library(timeSeries)
library(data.table)

library(plotly)
library(dplyr)
library(tidyr)

library(tidyverse)
library(plotly)


#######################################################################
# We initially applied an equally weighted buy-hold approach and hierarchical risk parity approach to the problem. However, these approaches produced results which did not meet the weighting constraints of the challenge..
# 
# We then planned to simplify our approach by focusing on the top n=109 performers from the training window and then apply a weighting of 1/n to each of these shares to keep within the constraints of the challenge. We created a matrix following the dimensions of the output weighting data frame and added the new weightings. This is ofcourse a naive approach given the time we spent working on the equally weighted buy-hold and hierarchical risk parity approaches. 
#######################################################################


# Read in the data --------------------------------------------------------
returns_train <- read_csv("data/returns_train.csv")
returns_test <- read_csv("data/returns_test.csv")


# Functions ---------------------------------------------------------------

# We need a function to generate a dataframe/tibble of equal weights. This
# "weights" dataframe must have the same dimensions as the returns dataframe.
# Each weight across each month is calculated as 1/p, where p is the number
# of stocks in the sample.

#' A function to generate equal weights
#'
#' @param data A dataframe, tibble or data.table
#'
#' @return same as input
equalise_weights <- function(data){
        
        data |>
                pivot_longer(contains("Stock")) |>
                group_by(month_end) |>
                mutate(value = 1/n()) |>
                ungroup() |>
                pivot_wider()
        
}



##################################################################################### 
####################### PORTFOLIO FUNCTION BEGINS  ##################################
##################################################################################### 

#' A function to generate your portflio
#'
#' Function to generate stocks weight allocation for time t+1
#' using historic data. Initial weights generated as 1/p for
#' active stock within a month
#'
#' @param training_data A data.frame, tibble or data.table
#' @param test_data A data.frame, tibble or data.table
#'
#' @return Same as input
generate_portfolio <- function(training_data, test_data){
        
        message(paste("Portfolio training data ranges from", min(training_data$month_end), 
                      "to", max(training_data$month_end)))
        
        message(paste("Portfolio test data ranges from", min(test_data$month_end),
                      "to", max(test_data$month_end)))
        
        
        
        
  ##################################################################################### 
        # YOUR CODE GOES BELOW THIS LINE ------------------------------------------
  ######################################################################################      
       
        ## Combine to run cummulative returns on later
        full <- bind_rows(returns_train, returns_test)
        # timeseries objedct to calcs
        tsRet <- timeSeries(full[,2:ncol(full)], as.Date(full[[1]]))
        
        ########## USES TRAINIG DATA ################
        
        # TS object for train set to calc top 10 
        Train_tsRet <- timeSeries(returns_train[,2:ncol(returns_train)], as.Date(returns_train[[1]]))
        
        ret_dim <- dim(Train_tsRet)
        ## 1. Compute the Geometric mean
        # Expected return for each asset in TRIANING set
        mG <- c()
        for (asset in 1:ret_dim[2]){
                mG[asset] <- exp(mean(log(Train_tsRet[,asset]+1)))-1 # n x 1 (where n = # assets)
        }
        
        
        ### Select top 10 from train based on cummulative geometric ret
        sorted_indices <- order(mG, decreasing = TRUE)
        
        # Select the last 10 indices (top 10 values)
        top_10_indices <- sorted_indices[1:10]
        
        ### Create storage for weights
        weight_store <- as.data.frame(matrix(0, nrow(tsRet), ncol(tsRet)))
        
        # Set the columns specified in 'top_10_indices' to 1/10 in 'weight_store'
        for (asset in top_10_indices) {
                weight_store[, asset] <- 0.1
        }
        
        # Top 10 assets have 1/N weights, all else = 0
        
        ### NOW each of the top 10 weights have weight of 0.1 at each timepoint
        print(weight_store[1,])
        
        ################ Below is a backtest for loop indexed through time
        
        ### BACKTEST LOOP SET UP:
        ########################################################################
        
        ###### initialize storage and inputs:
        N <- dim(tsRet)
        numYrs <- N[1]*12
        Window  <- 36 # (Can change between 3 to 4 years: 36 -> 48)
        
        
        # Step forwards by a month using loop
        tot <- N
        
        #1. Equally Weighted
        Overlap_tsERet        <- tsRet[,1]*0
        names(Overlap_tsERet) <- "Equally Weighted CM"
        
        
        #5. Constant Mix Port
        # Initialise weights
        df_weights <- weight_store
        Overlap_tsCMRet <-  tsRet[,1]*0  # CM Returns per rolled month
        names(Overlap_tsCMRet) <- "Port"
        
        
        for (i in Window:(tot[1]-1)){
                
                
                #### Call opt using above inputs, store returned wts
                #1.  Equal
                EWts <- rep(round(1/N[2],4), length.out = N[2])
                # 
                
                #2. Constant Mix
                # Weights calculated outside of for loop
                
                #### Calc + store realised returns, wts * actual market returns
                
                # #1. Equally weighted realised returns
                Overlap_tsERet[i] <- EWts %*% t(tsRet[i,])
                # # 
        }
        
        ##### CALCULATE REturns based off  full data set
        returns_matrix <- as.matrix(tsRet)
        resulting_matrix <- df_weights * returns_matrix
        
        # Step 2: Sum across rows to get the resulting vector
        resulting_vector <- rowSums(resulting_matrix)
        
        
        ######### FULL DATASET 
        df_returns <- timeSeries(resulting_vector, as.Date(full[[1]]))
        Cummulative <- 100*cumprod((df_returns)+1)
        tsCummulative <- timeSeries(Cummulative, as.Date(full[[1]]))
        
        df_returns <- as.matrix(df_returns)
        
        
 ##################################################################################### 
# YOUR CODE GOES ABOVE THIS LINE ------------------------------------------
 ##################################################################################### 
        
        
        
        # Your final weights dataframe should have a month_end column, followed by
        # columns of stocks each containing a weight for each date. e.g:
        #
        #   month_end   Stock1 Stock10  Stock11  Stock14 Stock16  Stock18   Stock19
        # 1 2010-01-31  0.0426 -0.0761 -0.150   -0.0313  -0.0530 -0.114   -0.0717
        # 2 2010-02-28 -0.0150 -0.100  -0.0233   0.00248  0.0771  0.0108   0.0224
        # 3 2010-03-31  0.112   0.0979  0.122    0.0644   0.107   0.0179   0.000892
        # 4 2010-04-30 -0.0405 -0.0343 -0.0328  -0.0302   0.0224  0.0855  -0.00716
        # 5 2010-05-31 -0.0694 -0.0732  0.00435  0.0108  -0.0255 -0.00189 -0.0125
        # 6 2010-06-30  0.0588  0.0310 -0.0669   0.0633  -0.0286  0.0203  -0.0644
        
        # We will use only the weights from the test set's earliest date and tack on
        # equal weights from the training set for charting purposes
        data_out <-
                training_data |>
                equalise_weights() |> 
                bind_rows(df_weights) |>
                arrange(month_end)
        
        # 10% limit check
        if(any(data_out[-1] > 0.101)) stop("One of your weights exceeds the 0.1 limit.")
        
        return(data_out)
        
}


#' Plot the total return
#' 
#' Uses the output from generate_portfolio() 
#'
#' @param df_returns a dataframe of returns
#' @param df_portfolio_weights a dataframe of portfolio weights
#' @param return_data 
#'
#' @return
#' plotly html widget
plot_total_return <- function(df_returns, df_portfolio_weights, return_data = FALSE) {
        
        returns_long <- pivot_longer(df_returns, contains("Stock"), values_to = "return")
        
        # generate equal weighted benchmark index
        benchmark_return <-
                equalise_weights(df_returns) |>
                pivot_longer(contains("Stock"), values_to = "weight") |>
                left_join(returns_long, by = c("month_end", "name")) |> 
                mutate(indexed = weight*return) |> 
                summarise(benchmark_return = sum(indexed), .by = c("month_end")) |> 
                mutate(benchmark_return = cumprod(1 + benchmark_return)*100) |> 
                pivot_longer(benchmark_return)
        
        # process the portfolio weight returns
        portfolio_return <-
                df_portfolio_weights |>
                pivot_longer(contains("Stock"), values_to = "weight") |> 
                left_join(returns_long, by = c("month_end", "name")) |> 
                mutate(indexed = weight*return) |>
                summarise(portfolio_return = sum(indexed), .by = c("month_end")) |> 
                mutate(portfolio_return = cumprod(1 + portfolio_return)*100) |> 
                pivot_longer(2)
        
        chart_data <- 
                bind_rows(benchmark_return, portfolio_return) |>
                mutate(name = stringr::str_to_title(gsub(pattern = "_", replacement = " ", x = name)))
        
        if (return_data) return(chart_data)
        
        plot <- 
                chart_data |>
                group_by(name) |>
                ggplot(aes(x = month_end, y = value, group = name, colour = name)) +
                geom_line() +
                labs(y = "Total Return", x = "Month End")
        
        ggplotly(plot)
        
}

# Run the solution --------------------------------------------------------

returns <- bind_rows(returns_train, returns_test)
portfolio_weights <- generate_portfolio(returns_train, returns_test)
plot_total_return(returns, portfolio_weights, return_data = FALSE)






plot(tsCummulative,plot.type = "s", col = c("orange", "magenta") ,lwd = c(2,2),at = "chic", format = "%Y %b", xlab = "Time", ylab = "Cummulative Returns")
# text(as.POSIXct("2009-02-28"),3.4,"Global Financial Crisis",pos = 2, cex = 1.1, srt = 90)
# text(as.POSIXct("2014-01-31"),4.9,"QE tappering",pos = 2, cex = 1.1, srt = 90)
# text(as.POSIXct("2012-08-31"),4.9,"Marikana massacre",pos = 2, cex = 1.1, srt = 90)
# text(as.POSIXct("2015-07-31"),2.5,"Anitdumping Policy",pos = 2, cex = 1.1, srt = 90)
## Add event lines
# abline(v = as.POSIXct("2009-02-28"),lwd = 3,col = "black") #GFC
# abline(v = as.POSIXct("2014-01-31"),lwd = 3,col = "black") # QE tappering
# abline(v = as.POSIXct("2012-08-31"), lwd = 3,col = "black") # Marikana massacre
# abline(v = as.POSIXct("2015-07-31"), lwd = 3,col = "black") # Antidumping
# title
title(main = "Overlapping Rolling Window Equity Curve")
# legend
# EQW, SR, BH, HRP, CM
legend("topleft",c(names(O_tsIndx)),col = c("orange","magenta"), lwd = c(2,2), lty = c('solid', 'solid'), bty = "o")
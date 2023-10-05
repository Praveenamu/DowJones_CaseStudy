# DowJones_CaseStudy
Machine Learning Model
title: "dowJones"
author: "PraveenaMunnam."

## Reading the dataset
```{r}
dow = read.csv("dow_jones_index.data", header = TRUE)
```

## Structure of the dataset

```{r}
str(dow)
```

## Cleaning the dataset

```{r}
dow$quarter <- as_factor(dow$quarter)
dow$stock <- as_factor(dow$stock)
dow$date <- as.Date(dow$date, "%m/%d/%Y")
dow$open <- as.numeric(str_remove(dow$open, "[$]"))
dow$high <- as.numeric(str_remove(dow$high, "[$]"))
dow$low <- as.numeric(str_remove(dow$low, "[$]"))
dow$close <- as.numeric(str_remove(dow$close, "[$]"))
dow$next_weeks_open <- as.numeric(str_remove(dow$next_weeks_open, "[$]"))
dow$next_weeks_close <- as.numeric(str_remove(dow$next_weeks_close, "[$]"))
```


## Structure of the dataset
```{r}
str(dow)
```

## Checking whether the data set have the null values

```{r}
colSums(is.na(dow))
```
### To handle these missing values, we grouped the data by stock and replaced the missing observations with the mean value of the respective variable for each stock.

```{r}
dow = dow %>% 
  group_by(stock) %>%
  mutate(percent_change_volume_over_last_wk = ifelse(is.na(percent_change_volume_over_last_wk),
                                                     mean(percent_change_volume_over_last_wk,
                                                     na.rm=TRUE),
                                                     percent_change_volume_over_last_wk),
         previous_weeks_volume = ifelse(is.na(previous_weeks_volume),
                                        mean(previous_weeks_volume, na.rm=TRUE), 
                                        previous_weeks_volume)) %>% 
  ungroup()
```

## Checking the data set

```{r}
sum(is.na(dow))
```

```{r}
names(dow)
```
# Exploratory Analysis

```{r}
dow %>%
  ggplot(aes(x = as.Date(date, format = "%m/%d/%Y"), y = percent_change_next_weeks_price)) +
  geom_line() +
  geom_point() +
  facet_wrap(~stock) +
  labs(title = "Percent Change Next Weeks Price")
```

## Check for correlation

```{r}
corrplot(cor(dow[,-(1:3)]))
```



## Interpretation:
Open, High, Low, Close, next_weeks_open, next_weeks_close.

# Overview of the data distribution

## Computing scatter plots for each variable


```{r}
pairs(dow)
```

## scatter plot of the percentage change in price of a stock against the duration of time from January to June 
```{r}
plot(dow$percent_change_next_weeks_price, ylab = "the percentage change in price of the stock", xlab = "Duration ( Jan to June)", pch = ".")
```



## Variables removed next_weeks_open, next_weeks_close,stock,percent_return_next_dividend

## Selection of the  variables

```{r}
lr = lm(percent_change_next_weeks_price ~ ., data = dow)
summary(lr)
```


```{r}
lr1 = lm(percent_change_next_weeks_price ~ quarter + date+ open+ high+ low + close+ volume + percent_change_price+ percent_change_volume_over_last_wk+ previous_weeks_volume + days_to_next_dividend, data = dow)
summary(lr1)
```

```{r}
r2 = step(lr1, direction = "backward")
```


```{r}
summary(r2)
```
## Lag variables

### Open variable

```{r}
lag.plot(dow$open, pch = ".", set.lags = 1:4)
```

```{r}
lag.plot(dow$high, pch = ".", set.lags = 1:4)
```


```{r}
lag.plot(dow$close, pch = ".", set.lags = 1:4)
```
```{r}
dow.lag <- dow %>% 
  group_by(stock) %>% 
  mutate(close.lag = lag(close, n = 1), 
         open.lag = lag(open, n = 1),
         high.lag = lag(high, n = 1)) %>% 
  ungroup()
```


```{r}
colSums(is.na(dow.lag))
```
```{r}
dow.lag <- dow.lag %>% 
  group_by(stock) %>%
  mutate(close.lag = ifelse(is.na(close.lag), mean(close.lag, na.rm=T), close.lag),
         open.lag = ifelse(is.na(open.lag), mean(open.lag, na.rm=T), open.lag),
         high.lag = ifelse(is.na(high.lag),
                                             mean(high.lag, na.rm=T),
                                             high.lag)) %>% 
  ungroup()
```

```{r}
colSums(is.na(dow.lag))
```

```{r}
dow.train = dow.lag[dow.lag$quarter == 1,]
dow.test = dow.lag[dow.lag$quarter == 2,]
```


## Linear Regression
```{r}
stocks = (unique(dow.train$stock))   ##  stocks
rmse = rep(NA, length(stocks))      ## root mean square error for each stock

dates = unique(dow.test$date)        # dates for predictions
Preds <- rep(NA, length(stocks))   ##dates for predictions

lm_predictions = data.frame(matrix(NA, ncol = 30, nrow = 13)) # df to store prediction of 30 stocks, 13 weeks
colnames(lm_predictions)= stocks
lm_pred_metrics = data.frame(Stock = stocks, rmse = rmse)   # df to store metrics

for(i in 1:length(stocks)){
  
  stock_train = subset(dow.train, stock == stocks[i]) # training  set for each stocks
  stock_test = subset(dow.test, stock == stocks[i])    # testing set for each stocks
                                                            # fit linear regression for each stocks
  lm_fit = lm(percent_change_next_weeks_price ~  volume + close.lag + high.lag  + open.lag,  
               data = stock_train) 
  
  lm_preds <-  predict(lm_fit, stock_test)                          # make predictions
  lm_predictions[i] <- lm_preds                   # store predictions for each week for each stocks
  
  lm_rmse = rmse(stock_test$percent_change_next_weeks_price, lm_preds)   # compute errors
  lm_pred_metrics[i,"rmse"] = lm_rmse                                    # store rmses

}
lm_predictions = data.frame(dates,lm_predictions)
# store predictions 
```
```{r}
lm_predictions
```

```{r}
models_rmse1 = data.frame(stocks, lm_pred_metrics$rmse)
models_rmse1
```

## SVM

```{r}
svm_pred_metrics <- data.frame(stocks, rmse)                        # df to store metrics

svm_predictions = data.frame(matrix(NA, ncol = 30, nrow = 13))      # df to store prediction of 30 stocks,                                                                       13 weeks
colnames(svm_predictions)= stocks


for(i in 1:length(stocks)){
  
  stock_train = subset(dow.train, stock == stocks[i])              # train set for each stocks
  stock_test = subset(dow.test, stock == stocks[i])                # test set for each stocks  
  set.seed(1)
                                                                    # fit SVM for each stocks
  tuned <- tune.svm(percent_change_next_weeks_price ~ volume + close + high  + open, 
                    data = stock_train,  gamma = seq(0.1, 1, by = 0.1), 
                    cost = seq(0.1,1, by = 0.1), scale=TRUE)
  
  svm_fit <- svm(percent_change_next_weeks_price ~ volume + close + high  + open, 
                   data = stock_train,  gamma = tuned$best.parameters$gamma,
                   cost = tuned$best.parameters$cost, scale=TRUE) 

  svm_preds <- predict(svm_fit, stock_test)                         # make prediction
  svm_predictions[i] <- svm_preds                      # store SVM predictions
  
  svm_rmse <- rmse(stock_test$percent_change_next_weeks_price, svm_preds)    # compute errors
  svm_pred_metrics[i, "rmse"] <- svm_rmse                                    # store rmse
}
```
```{r}
models_rmse2 = data.frame(stocks, svm_pred_metrics$rmse)
models_rmse2
```



## Decision tree
```{r}
dt_pred_metrics = data.frame(Stock = stocks, rmse = rmse)           # df to store metrics

for(i in 1:length(stocks)){

  stock_train = subset(dow.train, stock == stocks[i])              # train set for each stocks
  stock_test = subset(dow.test, stock == stocks[i])                # test set for each stocks
                                                                    # fit decision tree for each stocks 
  dt_model = tree(percent_change_next_weeks_price ~ volume + close.lag + high.lag  + open.lag, 
                  data = stock_train)  

  dt_preds = predict(dt_model, newdata = stock_test)                    # make predictions
  
  dt_rmse = rmse(stock_test$percent_change_next_weeks_price, dt_preds)  # compute errors
  dt_pred_metrics[i, "rmse"] = dt_rmse                                  # store rmse
}
```


# Plot the tree 
```{r}
plot(dt_model)
text(dt_model)
```


```{r}
models_rmse3 = data.frame(stocks, dt_pred_metrics$rmse)
models_rmse3
```


## Computing errors

```{r}
decision = mean((dt_preds - (dow.test$percent_change_next_weeks_price)) ^ 2)
decision
```
```{r}
SVMM = mean((svm_preds - (dow.test$percent_change_next_weeks_price)) ^ 2)
SVMM
```

```{r}
LinearM = mean((lm_preds - (dow.test$percent_change_next_weeks_price)) ^ 2)
LinearM
```

## Model comparision
```{r}
models_rmse = data.frame(stocks, lm_pred_metrics$rmse, dt_pred_metrics$rmse, svm_pred_metrics$rmse)
colnames(models_rmse)= c("stock", "lm_rmse", "dt_rmse", "svm_rmse")
colMeans(models_rmse[,2:4])
```


## CAPM

```{r}
# Compute Dow's values from the 30 stocks included in the dow industrial average using the Dow divisor for quarter 2

dow <- aggregate(dow.test$close, by = list(dow.test$date), FUN = function(x) sum(x)/0.132)   
return_dow <- na.omit(Delt(dow[,2]))

return_stocks <- data.frame(matrix(0.0, ncol = 30, nrow = 12))                        # df to store return
colnames(return_stocks) = c("AA", "AXP", "BA", "BAC", "CAT", "CSCO", "CVX", "DD", 
                            "DIS", "GE", "HD", "HPQ", "IBM", "INTC", "JNJ", "JPM", 
                            "KRFT", "KO", "MCD", "MMM", "MRK", "MSFT", "PFE", "PG", 
                            "T", "TRV", "UTX", "VZ", "WMT", "XOM")

all_Stocks =svm_predictions %>% pivot_longer(cols = 1:30,         # represent prediction in long format
                      names_to = "stock", values_to = "return")

for(i in 1:length(stocks)){                                       # compute returns 
  
  dow.sub = subset(dow.test, stock == stocks[i])
  return_stocks[i] = na.omit(Delt(dow.sub$close)) 
}

return_stocks <- data.frame(return_stocks, return_dow) %>% 
  rename(DOW = Delt.1.arithmetic)              # compute average returns for each stock. 
```

```{r}
returns_stk = t(return_stocks[,-31] %>% summarise(across(where(is.numeric), mean)))
colnames(returns_stk) = "return"
returns_stk= as.data.frame(returns_stk)
```


## Calculate beta for the stocks

```{r}
beta_AA = lm(AA ~ DOW, data = return_stocks)$coef[2]
beta_AXP = lm(AXP ~ DOW, data = return_stocks)$coef[2]
beta_BA = lm(BA ~ DOW, data = return_stocks)$coef[2]
beta_BAC = lm(BAC ~ DOW, data = return_stocks)$coef[2]
beta_CAT = lm(CAT ~ DOW, data = return_stocks)$coef[2]
beta_CSCO = lm(CSCO ~ DOW, data = return_stocks)$coef[2]
beta_CVX = lm(CVX ~ DOW, data = return_stocks)$coef[2]
beta_DD = lm(DD ~ DOW, data = return_stocks)$coef[2]
beta_DIS = lm(DIS ~ DOW, data = return_stocks)$coef[2]
beta_GE = lm(GE ~ DOW, data = return_stocks)$coef[2]
beta_HD = lm(HD ~ DOW, data = return_stocks)$coef[2]
beta_HPQ = lm(HPQ ~ DOW, data = return_stocks)$coef[2]
beta_IBM = lm(IBM ~ DOW, data = return_stocks)$coef[2]
beta_INTC = lm(INTC ~ DOW, data = return_stocks)$coef[2]
beta_JNJ = lm(JNJ ~ DOW, data = return_stocks)$coef[2]
beta_JPM = lm(JPM ~ DOW, data = return_stocks)$coef[2]
beta_KRFT = lm(KRFT ~ DOW, data = return_stocks)$coef[2]
beta_KO = lm(KO ~ DOW, data = return_stocks)$coef[2]
beta_MCD = lm(MCD ~ DOW, data = return_stocks)$coef[2]
beta_MMM = lm(MMM ~ DOW, data = return_stocks)$coef[2]
beta_MRK = lm(MRK ~ DOW, data = return_stocks)$coef[2]
beta_MSFT = lm(MSFT ~ DOW, data = return_stocks)$coef[2]
beta_PFE = lm(PFE ~ DOW, data = return_stocks)$coef[2]
beta_PG = lm(PG ~ DOW, data = return_stocks)$coef[2]
beta_T = lm(`T` ~ DOW, data = return_stocks)$coef[2]
beta_TRV = lm(TRV ~ DOW, data = return_stocks)$coef[2]
beta_UTX = lm(UTX ~ DOW, data = return_stocks)$coef[2]
beta_VZ = lm(VZ ~ DOW, data = return_stocks)$coef[2]
beta_WMT = lm(WMT ~ DOW, data = return_stocks)$coef[2]
beta_XOM = lm(XOM ~ DOW, data = return_stocks)$coef[2]

df_capm = data.frame(Stock = c("AA", "AXP", "BA", "BAC", "CAT", "CSCO", "CVX", "DD", "DIS", "GE",
                           "HD", "HPQ", "IBM", "INTC", "JNJ", "JPM", "KRFT", "KO", "MCD", "MMM",
                           "MRK", "MSFT", "PFE", "PG", "T", "TRV", "UTX", "VZ", "WMT", "XOM"),
                 Beta = c(beta_AA, beta_AXP, beta_BA, beta_BAC, beta_CAT, beta_CSCO,
                          beta_CVX, beta_DD, beta_DIS, beta_GE, beta_HD, beta_HPQ, beta_IBM,
                          beta_INTC, beta_JNJ, beta_JPM, beta_KRFT, beta_KO, beta_MCD,
                          beta_MMM, beta_MRK, beta_MSFT, beta_PFE, beta_PG, beta_T, beta_TRV,
                          beta_UTX, beta_VZ, beta_WMT, beta_XOM)) 
df_capm <- data.frame(df_capm, Return =returns_stk$return)
df_capm %>% 
  arrange(-desc(Beta)) %>% 
  mutate(Return = scales::percent(Return))
```

```{r}
return_stocks %>% pivot_longer(cols = 1:30, names_to = "stock", values_to = "return") %>%
  ggplot(aes(x = return, y = fct_rev(stock))) +
  geom_boxplot() + 
  scale_x_continuous(labels = scales::percent_format()) +
  labs(title = "Dow Jones Stock Return Distributions \n", x = "\n Expected Return", y = "Stock\n")+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  coord_flip()
```




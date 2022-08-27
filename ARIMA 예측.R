# load data
data<-read.csv("bigcon_data_1.csv", header=T, sep=",",encoding="UTF-8")
names(data)<-c("case","year", "month", "day", "hour", "water", "average", "area_A_rainfall", 
		"area_B_rainfall", "area_C_rainfall", "area_D_rainfall", 
		"area_E_height","area_D_height")

head(data)
y<-data[1:2891,6]
y<-data.frame(y)$y
plot(y, type="l")

## 분산안정화
log.y<-log(y)
plot(log.y, type="l")

## 차분 검토
acf(log.y, lag=40)
pacf(log.y, lag=40)

## modeling
library(forecast)
#log.y.ts<-as.ts(log.y, ts.frequency=1)
log.y.fit<-auto.arima(log.y, test = "adf", seasonal.test = "seas")

fit1<-Arima(log.y, order=c(3,0,1), seasonal=list(order=c(2,1,1), method="ML"))
fit1

acf(fit1$residuals)
pacf(fit1$residuals)

## 모형 검진
library(portes)
portest(log.y.fit$residuals, lags=c(6,12,18,24), test="LjungBox")

acf(log.y.fit$residuals)
pacf(log.y.fit$residuals)

portest(fit1$residuals, lags=c(6,12,18,24), test="LjungBox")

acf(fit1$residuals)
pacf(fit1$residuals)

##log series forecasting
log.hat<-forecast(log.y.fit, h=160) 
log.hat
plot(log.hat)
log.hat$upper
log.hat$lower

##original series forecasting
correction.term=exp(log.y.fit$sigma2/2)
y.hat.fit=exp(log.hat$mean)*correction.term
y.hat.upper=exp(log.hat$upper)*correction.term
y.hat.lower=exp(log.hat$lower)*correction.term
y.hat=cbind(y.hat.fit, y.hat.upper, y.hat.lower) #일단 예측 성공. 썩 그럴싸 한지는 잘 모르겠음;
plot(y.hat, type="l")

write.csv(y.hat,'forecast.csv')






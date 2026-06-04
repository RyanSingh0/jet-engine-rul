# Load necessary library for data manipulation
library(dplyr)
library(corrplot)
library(glmnet)
library(pROC)
library(reshape2)
library(ggplot2)

# Read training data: 100 engines with cycle data and 21 sensor measurements
train <- read.table("train_FD001.txt", header=FALSE)
# Assign column names: Unit, Cycle, 3 operational settings, Sensor1 to Sensor21
colnames(train) <- c("Unit","Cycle","Op1","Op2","Op3", paste0("Sensor", 1:21))

# Read test data and RUL (Remaining Useful Life) for FD001
test <- read.table("test_FD001.txt", header=FALSE)
colnames(test) <- colnames(train)
RUL <- scan("RUL_FD001.txt")  # vector of final RUL values for each engine in test

head(train)
summary(train)

# Check for missing values
sum(is.na(train))
sum(is.na(test))

set.seed(123)  # for reproducible sampling
train_sample <- train %>% sample_n(1000)
dim(train_sample)

# Compute the max cycle (life) for each engine in the sample
max_cycle <- train_sample %>% group_by(Unit) %>% summarize(maxCycle = max(Cycle))
# Merge maxCycle back to the sample
train_sample <- train_sample %>% left_join(max_cycle, by="Unit")
# Compute RUL: difference between maxCycle and current Cycle
train_sample <- train_sample %>%
  mutate(RUL = maxCycle - Cycle,
         FailSoon = as.factor(ifelse(RUL <= 30, "Yes", "No")))

# Remove the helper column maxCycle
train_sample$maxCycle <- NULL

head(train_sample)


# Summary statistics for RUL and a selected sensor (Sensor11 as example)
summary(train_sample$RUL)
summary(train_sample$Sensor11)

# Histogram of RUL
hist(train_sample$RUL, breaks=20, main="Histogram of RUL", xlab="Remaining Useful Life (cycles)")

# Histogram of Sensor11 values
hist(train_sample$Sensor11, breaks=20, main="Histogram of Sensor11", xlab="Sensor11")

# Scatterplot of RUL vs Sensor11
plot(train_sample$Sensor11, train_sample$RUL,
     main="RUL vs Sensor11", xlab="Sensor11", ylab="RUL")
# Add a smoothing line
lines(lowess(train_sample$Sensor11, train_sample$RUL), col="blue")

# Boxplot of Sensor11 by FailSoon group
boxplot(Sensor11 ~ FailSoon, data=train_sample,
        main="Sensor11 by FailSoon Group", xlab="FailSoon", ylab="Sensor11")


# Compute correlations of each sensor with RUL
sensor_cols <- paste0("Sensor", 1:21)
cors <- sapply(train_sample[sensor_cols], function(x) cor(x, train_sample$RUL))
# Display sensors with highest absolute correlation
cor_sorted <- sort(cors, decreasing=TRUE)
cor_sorted
corr_matrix <- cor(train_sample[, sensor_cols])
corrplot(corr_matrix, method="color", type="upper", t1.cex = 0.8, col=colorRampPalette(c("blue","white","red"))(200),
         title="Correlation Heatmap of Sensors", mar=c(0,0,1,0))
cors


mean_summary <- train_sample %>%
  group_by(Stage) %>%
  summarise(
    Mean_Sensor11 = round(mean(Sensor11, na.rm = TRUE), 2),
    Mean_Sensor12 = round(mean(Sensor12, na.rm = TRUE), 2),
    Count = n()
  )

mean_summary
str(train_sample)
# One-samplstage()# One-sample t-test: H0: mean(Sensor11) = 0 / baseline
t_one <- t.test(train_sample$Sensor11, mu = 47.34)
t_one


# Two-sample t-test for Sensor11 between FailSoon groups
sensor11_FailYes <- subset(train_sample, FailSoon=="Yes")$Sensor11
sensor11_FailNo  <- subset(train_sample, FailSoon=="No")$Sensor11

t_two <- t.test(sensor11_FailYes, sensor11_FailNo)
t_two


# Create RUL quartile groups
train_sample <- train_sample %>%
  mutate(RUL_Group = cut(RUL, breaks=quantile(RUL, probs = seq(0,1,0.25)),
                         include.lowest=TRUE, labels=c("Q1","Q2","Q3","Q4")))

# One-way ANOVA: Sensor11 ~ RUL_Group
anova1 <- aov(Sensor11 ~ RUL_Group, data=train_sample)
summary(anova1)

# Tukey HSD post-hoc test
TukeyHSD(anova1)

# Define Cycle Stage by each engine’s life (1/3 portions)
train_sample <- train_sample %>%
  group_by(Unit) %>%
  mutate(TotalCycles = max(Cycle),
         Stage = case_when(
           Cycle <= TotalCycles/3       ~ "Early",
           Cycle <= 2*TotalCycles/3     ~ "Mid",
           TRUE                         ~ "Late"
         )) %>%
  ungroup()

# Convert factors
train_sample$Unit <- as.factor(train_sample$Unit)
train_sample$Stage <- factor(train_sample$Stage, levels=c("Early","Mid","Late"))

# Two-way ANOVA: RUL ~ Unit * Stage
anova2 <- aov(RUL ~ Unit * Stage, data=train_sample)
summary(anova2)


# Compute total life per engine (from full training data)
total_life <- train %>% group_by(Unit) %>% summarize(MaxCycle = max(Cycle))
median_life <- median(total_life$MaxCycle)

# Label engines as short-life or long-life
total_life <- total_life %>%
  mutate(LifeGroup = ifelse(MaxCycle <= median_life, "Short", "Long"))


train_sample$Unit <- as.integer(as.character(train_sample$Unit))
total_life$Unit <- as.integer(total_life$Unit)

# Merge LifeGroup into sample
train_sample <- train_sample %>%
  left_join(total_life %>% select(Unit, LifeGroup), by="Unit")

train_sample$LifeGroup <- as.factor(train_sample$LifeGroup)

# ANCOVA: Sensor11 ~ Cycle * LifeGroup
ancova_model <- lm(Sensor11 ~ Cycle * LifeGroup, data=train_sample)
summary(ancova_model)


# Fit linear model: RUL as function of Sensor11
lm_w <- lm(RUL ~ Cycle, data = train_sample)
summary(lm_w)
lm_simple <- lm(RUL ~ Sensor11, data=train_sample)
summary(lm_simple)

# Plot regression line
plot(train_sample$Sensor11, train_sample$RUL,
     main="Linear Regression: RUL ~ Sensor11", xlab="Sensor11", ylab="RUL")
abline(lm_simple, col="red", lwd=2)

# Multiple regression model
lm_multi <- lm(RUL ~ Sensor11 + Sensor9, data=train_sample)
summary(lm_multi)

# Calculate VIF to check multicollinearity
library(car)
vif(lm_multi)

# Residual analysis
par(mfrow=c(2,2))
plot(lm_multi)  # this produces Residuals vs Fitted, QQ, Scale-Location, Cook's distance
par(mfrow=c(1,1))

# Fit logistic regression: FailSoon (Yes/No) on Sensor11
logit_simple <- glm(FailSoon ~ Sensor11, data=train_sample, family=binomial)
summary(logit_simple)

# Compute and display odds ratio for Sensor11
exp(coef(logit_simple))

# Predict probabilities and classify (threshold = 0.5)
train_sample$PredProb1 <- predict(logit_simple, type="response")
train_sample$PredClass1 <- ifelse(train_sample$PredProb1 > 0.5, "Yes", "No")
train_sample$PredClass1 <- factor(train_sample$PredClass1, levels=c("No","Yes"))

# Confusion matrix and accuracy
table(Predicted = train_sample$PredClass1, Actual = train_sample$FailSoon)
mean(train_sample$PredClass1 == train_sample$FailSoon)

# Fit multiple logistic regression
logit_multi <- glm(FailSoon ~ Sensor11 + Sensor12, data=train_sample, family=binomial)
summary(logit_multi)

# Odds ratios
exp(coef(logit_multi))

# Predictions
train_sample$PredProb2 <- predict(logit_multi, type="response")
train_sample$PredClass2 <- ifelse(train_sample$PredProb2 > 0.5, "Yes", "No")
train_sample$PredClass2 <- factor(train_sample$PredClass2, levels=c("No","Yes"))

# Confusion matrix and accuracy
table(Predicted = train_sample$PredClass2, Actual = train_sample$FailSoon)
mean(train_sample$PredClass2 == train_sample$FailSoon)

# ROC curve (requires pROC or ROCR package)
roc_obj <- roc(train_sample$FailSoon, train_sample$PredProb2, levels=c("No","Yes"))
plot(roc_obj, main="ROC Curve for Logistic Model")
auc(roc_obj)

# Preprocess test: compute max test cycle per engine and assign final RUL
test_max <- test %>% group_by(Unit) %>% summarize(MaxCycle = max(Cycle))
test_max$FinalRUL <- RUL  # RUL vector is in engine order 1 to 100

# Merge FinalRUL to test data
test <- test %>% left_join(test_max, by="Unit")
# Compute RUL for each test cycle: remaining life = FinalRUL + (MaxCycle - Cycle)
test <- test %>%
  mutate(RUL = FinalRUL + (MaxCycle - Cycle))

# Predict RUL using multiple linear regression model
test$PredRUL <- predict(lm_multi, newdata=test)

# Evaluate regression: RMSE and R-squared-like measure on test
rmse <- sqrt(mean((test$PredRUL - test$RUL)^2))
correlation <- cor(test$PredRUL, test$RUL)
cat("Test RMSE:", rmse, "\n")
cat("Test Correlation (R):", correlation, "\n")

# Classify FailSoon in test
test$FailSoon <- as.factor(ifelse(test$RUL <= 30, "Yes", "No"))
test$PredProb1 <- predict(logit_simple, newdata=test, type="response")
test$PredClass1 <- ifelse(test$PredProb1 > 0.5, "Yes", "No")
test$PredClass1 <- factor(test$PredClass1, levels=c("No","Yes"))

# Classification evaluation
conf_mat <- table(Predicted = test$PredClass1, Actual = test$FailSoon)
accuracy <- mean(test$PredClass1 == test$FailSoon)
conf_mat
cat("Test Accuracy:", accuracy, "\n")

# ROC curve for test data
roc_test <- roc(test$FailSoon, test$PredProb1, levels=c("No","Yes"))
plot(roc_test, main="ROC Curve on Test Data")
auc(roc_test)

summary(train_sample)


# Prepare data
X <- as.matrix(train_sample[, sensor_cols])
y <- ifelse(train_sample$FailSoon == "Yes", 1, 0)

# Standardize
X_scaled <- scale(X)

# Fit regularized models using glmnet
set.seed(555)
# Remove sensors with all NAs
bad_sensors <- c("Sensor1", "Sensor5", "Sensor10", "Sensor16", "Sensor18", "Sensor19")
X <- train_sample[, setdiff(sensor_cols, bad_sensors)]


# Rebuild scaled matrix and response
X_scaled <- scale(as.matrix(X))
y <- ifelse(train_sample$FailSoon == "Yes", 1, 0)

# Proceed with glmnet
cv_lasso <- cv.glmnet(X_scaled, y, alpha=1, family="binomial")

cv_lasso <- cv.glmnet(X_scaled, y, alpha=1, family="binomial")  # Lasso
cv_ridge <- cv.glmnet(X_scaled, y, alpha=0, family="binomial")  # Ridge
cv_elnet <- cv.glmnet(X_scaled, y, alpha=0.5, family="binomial")  # Elastic Net

# ROC Curve
prob_lasso <- predict(cv_lasso, X_scaled, type="response", s="lambda.min")
prob_ridge <- predict(cv_ridge, X_scaled, type="response", s="lambda.min")
prob_elnet <- predict(cv_elnet, X_scaled, type="response", s="lambda.min")

roc_lasso <- roc(y, as.numeric(prob_lasso))
roc_ridge <- roc(y, as.numeric(prob_ridge))
roc_elnet <- roc(y, as.numeric(prob_elnet))

plot(roc_lasso, col="red", main="ROC Curve for Regularized Logistic Models")
lines(roc_ridge, col="blue")
lines(roc_elnet, col="darkgreen")
legend("bottomright", legend=c(
  paste("Lasso (AUC =", round(auc(roc_lasso), 3), ")"),
  paste("Ridge (AUC =", round(auc(roc_ridge), 3), ")"),
  paste("Elastic Net (AUC =", round(auc(roc_elnet), 3), ")")
), col=c("red", "blue", "darkgreen"), lty=1)

# Coefficient comparison
coefs <- data.frame(
  Sensor = rownames(coef(cv_lasso, s="lambda.min"))[-1],
  Lasso = as.numeric(coef(cv_lasso, s="lambda.min"))[-1],
  Ridge = as.numeric(coef(cv_ridge, s="lambda.min"))[-1],
  ElasticNet = as.numeric(coef(cv_elnet, s="lambda.min"))[-1]
)

coefs_melt <- melt(coefs, id.vars="Sensor", variable.name="Model", value.name="Coefficient")

# Barplot of coefficients
ggplot(coefs_melt, aes(x=Sensor, y=Coefficient, fill=Model)) +
  geom_bar(stat="identity", position="dodge") +
  coord_flip() +
  labs(title="Coefficient Comparison of Regularized Logistic Models") +
  theme_minimal()


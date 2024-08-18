library(lme4)
library(ggplot2)
library(MuMIn)

setwd("/Users/shine/Documents/MSc/Neuro\ Research/Ovarian_hormone/Dump/")
dataset = read.csv("hormone_data.csv")

dataset = read.csv("hormone_data.csv")
pairwise.wilcox.test(dataset$metric, dataset$phase, p.adjust.method = "BH")
boxplot(metric ~ phase, data=dataset)
means <- tapply(dataset$metric, dataset$phase, mean)
points(means, pch=20, cex=1.5)

hormone_pos.model_2 = lmer(metric ~ est + pro + est*pro  + (1|cycle), data=dataset, REML = FALSE)
hormone_pos.model = lmer(metric ~ est + pro + est*pro  + (1|cycle), data=dataset)
hormone_pos.pro_test = lmer(metric ~ est  + (1|cycle), data=dataset, REML = FALSE)
hormone_pos.est_test = lmer(metric ~ pro + (1|cycle), data=dataset, REML = FALSE)
hormone_pos.null = lmer(metric ~ (1|cycle), data=dataset, REML = FALSE)
anova(hormone_pos.null, hormone_pos.model_2)
anova(hormone_pos.pro_test, hormone_pos.model_2)
anova(hormone_pos.est_test, hormone_pos.model_2)


lem <- lmer(data = dataset, metric ~ est + (1 | cycle))
lpm <- lmer(data = dataset, metric ~ pro + (1 | cycle))
lmm <- lmer(data = dataset, metric ~ est + pro + (1 | cycle))
ogm <- lmer(data = dataset, metric ~ est + pro + est*pro + (1 | cycle))
models <- list(lem, lpm, lmm, ogm)
model.names <- c('lem', 'lpm', 'lmm', 'ogm')
aictab(cand.set = models, modnames = model.names)



lme_mod = lmer(data = dataset, metric ~ est + pro + est*pro + (1 | cycle))
plot(metric ~ pro, data = dataset)
points(dataset$pro[order(dataset$pro)], fitted(lme_mod)[order(dataset$pro)], pch = 19)

plot(metric ~ est, data = dataset)
points(dataset$est[order(dataset$est)], fitted(lme_mod)[order(dataset$est)], pch = 19)

dataset$z <- with(dataset, est + pro + est*pro)
plot(metric ~ z, data = dataset)
points(dataset$z[order(dataset$z)], fitted(lme_mod)[order(dataset$z)], pch = 19)


praw <-
  ggplot(data = dataset, aes(
    x = est,
    y = metric,
    group = cycle)
  )
praw + geom_point() + geom_line() 


bestm <- lmer(data = dataset, metric ~ pro + (1 | cycle), REML = FALSE)
null <- lmer(data = dataset, metric ~ (1 | cycle), REML = FALSE)
anova(null, bestm)


hormone_neg.model_2 = lmer(neg_con ~ est + pro + est*pro + (1|cycle), data=dataset, REML = FALSE)
hormone_neg.model = lmer(neg_con ~ est + pro + est*pro + (1|cycle), data=dataset)
hormone_neg.null_e = lmer(neg_con ~ est  + (1|cycle), data=dataset, REML = FALSE)
hormone_neg.null_p = lmer(neg_con ~ pro + (1|cycle), data=dataset, REML = FALSE)

# If progesterone is facilitating + Connectivity changes
anova(hormone_pos.null_e, hormone_pos.model_2)
anova(hormone_neg.null_e, hormone_neg.model_2)
# If Estrodial is facilitating + Connectivity changes
anova(hormone_pos.null_p, hormone_pos.model_2)
anova(hormone_neg.null_p, hormone_neg.model_2)

pairwise.wilcox.test(dataset$metric, dataset$phase, p.adjust.method = "BH")



boxplot(neg_con ~ phase, data=dataset)
means <- tapply(dataset$neg_con, dataset$phase, mean)
points(means, pch=20, cex=1.5)


#kruskal.test(pos_con ~ phase, data = dataset)
pairwise.wilcox.test(dataset$pos_con, dataset$phase, p.adjust.method = "BH")

hormone_pos.model_2 = lmer(pos_con ~ est  + (1|cycle), data=dataset, REML = FALSE)
hormone_pos.null_e = lmer(pos_con ~ (1|cycle), data=dataset, REML = FALSE)



summary(hormone_pos.model)
summary(hormone_neg.model)


# --- behavior
dataset = read.csv("hormone_data.csv")

behave.model_2 = lmer(d_prime ~ est + pro + est*pro  + (1|cycle), data=dataset, REML = FALSE)
behave.model = lmer(d_prime ~ est + pro + est*pro  + (1|cycle), data=dataset)
behave.null_e = lmer(d_prime ~ est  + (1|cycle), data=dataset, REML = FALSE)
behave.null_p = lmer(d_prime ~ pro + (1|cycle), data=dataset, REML = FALSE)

anova(behave.null_e, behave.model_2)
# If Estrodial is facilitating + Connectivity changes
anova(behave.null_p, behave.model_2)



# ------------- NON LINEAR
library(lme4)
library(ggplot2)

setwd("/Users/shine/Documents/MSc/Neuro\ Research/Ovarian_hormone/Dump/")
dataset = read.csv("hormone_data.csv")

## --- AIC to determine hormone model 
lem <- lmer(data = dataset, metric ~ est + (1 | cycle))
qem <- lmer(data = dataset, metric ~ poly(est,2) + (1 | cycle))
lpm <- lmer(data = dataset, metric ~ pro + (1 | cycle))
qpm <- lmer(data = dataset, metric ~ poly(pro,2) + (1 | cycle))
lmm <- lmer(data = dataset, metric ~ est + pro + (1 | cycle))
#qmm <- lmer(data = dataset, metric ~ poly(est,2) + poly(pro,2) + (1 | cycle))
intmm <- lmer(data = dataset, metric ~ est*pro + (1 | cycle))
ogm <- lmer(data = dataset, metric ~ est + pro + est*pro + (1 | cycle))
#xm <- lmer(data = dataset, metric ~ est + poly(pro,2) + est*pro + (1 | cycle))

models <- list(lem, qem, lpm, qpm, lmm, intmm, ogm)
model.names <- c('lem', 'qem', 'lpm', 'qpm', 'lmm', 'intmm', 'ogm')

aictab(cand.set = models, modnames = model.names)

# Test if model is more significant than random chance
bestm <- lmer(data = dataset, metric ~ pro + (1 | cycle), REML = FALSE)
null <- lmer(data = dataset, metric ~ (1 | cycle), REML = FALSE)
anova(null, bestm)


# Spaghetti plot of the raw longitudinal data (20 cycles x 3 time points (with missing data))
praw <-
  ggplot(data = dataset, aes(
    x = pro,
    y = metric,
    group = cycle)
  )
praw + geom_point() + geom_line() # + facet_grid(cols = vars(In$Sex))

## Fit LME Model 
lme_mod = lmer(data = dataset, interaction.strength ~ pro + est + est*pro + (1 | cycle))

## Spaghetti Plot of the fit longitudinal data (67 individuals x 2 time points)
pfit <-
  ggplot(data = dataset, aes(
    x = pro,
    y = fitted(lme_mod),
    group = cycle)
  )
pfit + geom_point() + geom_line() # + facet_grid(cols = vars(In$Sex))


plot(metric ~ pro, data = dataset)
points(dataset$pro[order(dataset$pro)], fitted(lme_mod)[order(dataset$pro)], pch = 19)



null.est_driven <- lmer(data = dataset, pos_con ~ pro + (1 | cycle), REML = FALSE)
null.pro_driven <- lmer(data = dataset, pos_con ~ est + (1 | cycle), REML = FALSE)
anova(null.est_driven, bestm)
anova(null.pro_driven, bestm)

# ---------------------
dataset = read.csv("hormone_data.csv")
pairwise.wilcox.test(dataset$interaction.strength, dataset$phase, p.adjust.method = "BH")
boxplot(interaction.strength ~ phase, data=dataset)
means <- tapply(dataset$interaction.strength, dataset$phase, mean)
points(means, pch=20, cex=1.5)

hormone_pos.model_2 = lmer(interaction.strength ~ est + pro + est*pro  + (1|cycle), data=dataset, REML = FALSE)
hormone_pos.model = lmer(interaction.strength ~ est + pro + est*pro  + (1|cycle), data=dataset)
hormone_pos.pro_test = lmer(interaction.strength ~ est  + (1|cycle), data=dataset, REML = FALSE)
hormone_pos.est_test = lmer(interaction.strength ~ pro + (1|cycle), data=dataset, REML = FALSE)
hormone_pos.null = lmer(interaction.strength ~ (1|cycle), data=dataset, REML = FALSE)
anova(hormone_pos.null, hormone_pos.model_2)
anova(hormone_pos.pro_test, hormone_pos.model_2)
anova(hormone_pos.est_test, hormone_pos.model_2)

lme_mod = lmer(data = dataset, interaction.strength ~ est + pro + est*pro + (1 | cycle))
plot(interaction.strength ~ pro, data = dataset)
points(dataset$pro[order(dataset$pro)], fitted(lme_mod)[order(dataset$pro)], pch = 19)

lme_mod = lmer(data = dataset, interaction.strength ~ est + pro + est*pro + (1 | cycle))
plot(interaction.strength ~ est, data = dataset)
points(dataset$est[order(dataset$est)], fitted(lme_mod)[order(dataset$est)], pch = 19)

pfit <-
  ggplot(data = dataset, aes(
    x = est,
    y = interaction.strength,
    group = cycle)
  )
pfit + geom_point() + geom_line() 



# --- connectivity
dataset = read.csv("hormone_data.csv")
pairwise.wilcox.test(dataset$node.strength, dataset$phase, p.adjust.method = "BH")
boxplot(node.strength ~ phase, data=dataset)
means <- tapply(dataset$node.strength, dataset$phase, mean)
points(means, pch=20, cex=1.5)

hormone_pos.model_2 = lmer(node.strength ~ est + pro + est*pro  + (1|cycle), data=dataset, REML = FALSE)
hormone_pos.model = lmer(node.strength ~ est + pro + est*pro  + (1|cycle), data=dataset)
hormone_pos.pro_test = lmer(node.strength ~ est  + (1|cycle), data=dataset, REML = FALSE)
hormone_pos.est_test = lmer(node.strength ~ pro + (1|cycle), data=dataset, REML = FALSE)
hormone_pos.null = lmer(node.strength ~ (1|cycle), data=dataset, REML = FALSE)
anova(hormone_pos.null, hormone_pos.model_2)
anova(hormone_pos.pro_test, hormone_pos.model_2)
anova(hormone_pos.est_test, hormone_pos.model_2)


lme_mod = lmer(data = dataset, node.strength ~ est + pro + est*pro + (1 | cycle))
plot(node.strength ~ pro, data = dataset)
points(dataset$pro[order(dataset$pro)], fitted(lme_mod)[order(dataset$pro)], pch = 19)

lme_mod = lmer(data = dataset, node.strength ~ est + pro + est*pro + (1 | cycle))
plot(node.strength ~ est, data = dataset)
points(dataset$est[order(dataset$est)], fitted(lme_mod)[order(dataset$est)], pch = 19)


pfit <-
  ggplot(data = dataset, aes(
    x = est,
    y = node.strength,
    group = cycle)
  )
pfit + geom_point() + geom_line() 





pfit <-
  ggplot(data = dataset, aes(
    x = pro,
    y = fitted(lme_mod),
    group = cycle)
  )
pfit + geom_point() + geom_line() 

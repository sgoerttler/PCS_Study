library(lme4)

# load the tabular data for unpaired trials from the entire test phase.
data_all <- read.csv(file = 'data/data_preprocessed/data_PCS_study_all.csv')

vpsych <- function(){
	## link
	linkfun <- function(y) {
		# Note: this is not the right link function, but the link function is only used for the initialization,
		# and this initialization works better
		qnorm((y - 0.0) / 1, mean = 0, sd = 1)
		}
	## inverse link
	linkinv <- function(eta) {
		pnorm(eta, mean = 0, sd = 1) * 0.70 + 0.25
		}
	## derivative of invlink wrt eta
	mu.eta <- function(eta) dnorm(eta) * 0.70
	valideta <- function() TRUE
	link <- "4AFC"
	structure(list(linkfun = linkfun, linkinv = linkinv,
				   mu.eta = mu.eta, valideta = valideta,
				   name = link),
			  class = 'link-glm')
}

# create the link function
vv <- vpsych()

# fit the psychometric curves to the data using a generalized linear mixed model
model_psy <- glmer(acc ~ bright + (1 | col) + (1 + bright | part), data=data_all, family=binomial(link = vv))
print(summary(model_psy))

write.csv(coef(summary(model_psy)), "data/glmer_psychometric_curves/coef_PCS_study.csv")
write.csv(ranef(model_psy), "data/glmer_psychometric_curves/ranef_PCS_study.csv")


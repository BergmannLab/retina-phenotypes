library(survival)
library(survminer)
library(lubridate)
library(comprehenr)
library(tidyverse)
library(ggplot2)
library(parallel)

# Cox model for all retinal phenotypes

# Uses traits that have previously been linearly corrected for covariate effects

# predicts mortality as well as the following diseases:
diseases = c("age_diabetes", "age_angina", "age_heartattack", "age_DVT", "age_stroke", "age_glaucoma", "age_cataract", "age_other_serious_eye_condition", "age_pulmonary_embolism")

# Covariates are sex, age, age2, eye geometry, and PC1-20

args=commandArgs(trailingOnly=TRUE)
RUNDIR=args[1] #"/NVME/decrypted/scratch/multitrait/UK_BIOBANK_ZERO"
PHENOTYPE_ID=args[2] #"2022_07_08_ventile5"
filter_instance=args[3] # FALSE

#RUNDIR="/NVME/decrypted/scratch/multitrait/UK_BIOBANK_ZERO"
#PHENOTYPE_ID="2022_07_08_ventile2"
#filter_instance=FALSE

covarfile=paste0(RUNDIR,"/diseases_cov/",PHENOTYPE_ID,"_diseases_cov.csv")
traitfile=paste0(RUNDIR,"/participant_phenotype/",PHENOTYPE_ID,"_corrected_z.csv")


if (filter_instance == TRUE) {
    outid=paste0(PHENOTYPE_ID,"__cox_instance_0")
} else {
    outid=paste0(PHENOTYPE_ID,"__cox")
}

outfile_significance = paste0(RUNDIR,"/diseases_cov/",outid,"_heatmap_w_significance.pdf")
outfile_heatmap = paste0(RUNDIR,"/diseases_cov/",outid,"_heatmap.pdf")
outfile_pval = paste0(RUNDIR,"/diseases_cov/",outid,"_pval.csv")
outfile_full = paste0(RUNDIR,"/diseases_cov/",outid,"_full_results.csv")


# function to score single cox trait
cox_single = function(j) {
    
    # current trait
    #print(j)
    
    #conservative model (includes SBP, DBP, PR, hair+skin colour)
    # cox_model = coxph( Surv(time=years_to_event, event=event) ~ get(j) + SBP + DBP + PR + hair_colour + skin_colour + age + age2 + sex + spherical_power + spherical_power_2 + cylindrical_power + cylindrical_power_2 + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + PC11 + PC12 + PC13 + PC14 + PC15 + PC16 + PC17 + PC18 + PC19 + PC20, data=df)
    
    # DEFAULT MODEL
    cox_model = coxph( Surv(time=years_to_event, event=event) ~ get(j) + age + age2 + sex + spherical_power + spherical_power_2 + cylindrical_power + cylindrical_power_2 + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + PC11 + PC12 + PC13 + PC14 + PC15 + PC16 + PC17 + PC18 + PC19 + PC20, data=df)

    
    # print(summary(cox_model))
    
    # print(summary(cox_model)$coefficients[1,])
    # print(exp(cox_model$coefficients[1]),as.numeric(summary(cox_model)$coefficients[1,][5]))
    
    # hr = append(hr, exp(cox_model$coefficients[1]))
    # pval = append(pval, as.numeric(summary(cox_model)$coefficients[1,][5]))
    # 
    # intervals=data.frame(confint(cox_model))
    # lower.025 = append(lower.025, exp(intervals$X2.5..[1]))
    # upper.975 = append(upper.975, exp(intervals$X97.5..[1]))
    
    hr = exp(cox_model$coefficients[1])
    pval = as.numeric(summary(cox_model)$coefficients[1,][5])
    
    intervals=data.frame(confint(cox_model))
    lower.025 = exp(intervals$X2.5..[1])
    upper.975 = exp(intervals$X97.5..[1])
    
    return(list('hr'=hr,'pval'=pval, 'lower.025'=lower.025,'upper.975'=upper.975))
}

# load covariates
df = read.csv(covarfile)
rownames(df) = df$eid
df = df[-1]

# removing nans
print(paste("NA rows in covar file:",sum(is.na(df$age_recruitment))))

tmp = tmp=df[is.na(df$age_recruitment),]
df = df[!row.names(df) %in% rownames(tmp),]

# load z-scored, corrected traits, merge w/ covar
traits = read.csv(traitfile)
rownames(traits) = traits$X
traits = traits[-1]
traitnames = colnames(traits)
df = merge(df,traits,by=0)

# option to only keep instance 0
# this increased Cox performance for Zekavat et al.
if (filter_instance==TRUE) {
    df = df[df$instance == 0, ]
}

# df[df == -999.000000] <- NaN
df[df == -Inf] <- NaN
df[df == Inf] <- NaN

df$sex = as.factor(df$sex)
df$age = df$age_center
df$age2 = df$age_center_2


# squared eye geometry terms
df$spherical_power_2 = df$spherical_power^2
df$cylindrical_power_2 = df$cylindrical_power^2


#####
# DISEASE COX MODEL LOOP
# time to event is the age of the participant
#####

dis_idx=0
for(i in diseases) {
        
    # current disease
    print(i)
    
    dis_idx=dis_idx+1
    
    age_disease = coalesce(df[[paste(i,"_00",sep="")]], df[[paste(i,"_10",sep="")]],df[[paste(i,"_20",sep="")]],df[[paste(i,"_30",sep="")]])
    df$age_disease = age_disease
    
    df$max_age = coalesce(df$age_death, df$age_center_20, df$age_center_10, df$age_center_00) # coalesce keeps first not nan, prioritize death
    
    df$years_to_event = coalesce(df$age_disease,df$max_age) # if no disease (age_disease=NA), years_to_event is set to last age observed
    hist(df$years_to_event)
    
    # event: 1, right-censored: 0
    event = ifelse(is.na(df$age_disease), 0, 1) # if na, 0 (no disease), else 1 (disease)
    df$event = event
    hist(event)
    hist(df$event) # plot event frequency
    
    df_disease = df
    df_disease[df_disease$years_to_event<0,] = NA # age_disease can be negative in UKBB: -1: do not know, -3: prefer not to answer. we censor these
    
    # Compute Cox in parallel for all traits
    out_i=mclapply(traitnames, cox_single, mc.cores=72)
    
    # process disease results
    out_i=lapply(out_i, unlist)
    hr = c()
    pval = c()
    lower.025 = c()
    upper.975 = c()
    for (j in out_i) {
        hr = append(hr, j[1])
        pval = append(pval, j[2])
        lower.025 = append(lower.025, j[3])
        upper.975 = append(upper.975, j[4])
    }
    
    
    # add disease results to full table
    if (dis_idx == 1) {
        col= paste(i,'_hr', sep='')
        out = data.frame(tmpcol = hr, row.names=traitnames)
        colnames(out) = col
    }else {
        out[paste(i,'_hr', sep='')] = hr
    }
    out[paste(i,'_pval',sep='')] = pval
    out[paste(i,'_lower.025',sep='')] = lower.025
    out[paste(i,'_upper.975',sep='')] = upper.975
}

#####
# MORTALITY COX MODEL
# time to event is difference between age at first visit and age at death
#####

i='age_death'
df$age_disease = df$age_death

df$years_to_event = df$age_disease - df$age_center # 0 at date of measurement

# needed by coxph
end_time = max(df$years_to_event, na.rm=T) + 0.01 # approximation define censor time as largest observed years_to_event interval
df$years_to_event[is.na(df$years_to_event)] = end_time
hist(df$years_to_event)

event = ifelse(is.na(df$age_disease), 0, 1) # if na, 0 (no disease), else 1 (disease)
df$event = event

# mortality cox
out_i=mclapply(traitnames, cox_single, mc.cores=72)

# process results
out_i=lapply(out_i, unlist)
hr = c()
pval = c()
lower.025 = c()
upper.975 = c()
for (j in out_i) {
    hr = append(hr, j[1])
    pval = append(pval, j[2])
    lower.025 = append(lower.025, j[3])
    upper.975 = append(upper.975, j[4])
}


# add mortality to results table
out[paste(i,'_hr', sep='')] = hr
out[paste(i,'_pval',sep='')] = pval
out[paste(i,'_lower.025',sep='')] = lower.025
out[paste(i,'_upper.975',sep='')] = upper.975


#####
# VISUALIZATION - MORTALITY & DISEASE
#####


cols = colnames(out)
pval_cols = to_vec(for(i in cols) if(grepl( 'pval', i, fixed = TRUE)) i)


plot_traits = c('DF_artery', 'DF_vein', 'tau2_longestFifth_artery', 'tau2_longestFifth_vein', 'bifurcations', 'slope', 'mean_taa_angle', 'mean_tva_angle', 'medianDiameter_longestFifth_artery', 'medianDiameter_longestFifth_vein', 'D_A_std_std', 'D_V_std_std')
plot_traits = traitnames

out_pval = out[plot_traits,pval_cols]
colnames(out_pval) = c('7. Diabetes','4. Angina','3. Heart attack', '5. Deep vein thrombosis', '2. Stroke','8.1. Glaucoma','8.2. Cataract','8.3. Other serious eye condition','6. Pulmonary embolism','1. Mortality')
# rownames(out_pval) = c('0. Arterial DF', '0.5 Venular DF', '1. Arterial tau2', '1.5. Venular tau2', '2. Bifurcations', '3. Fractal dimension', '4. Arterial termporal angle', '4.5. Venular temporal angle', '5. Arterial diameter', '5.5 Venular diameter', '6 Arterial diameter STD', '6.6. Venular diameter STD')
rownames(out_pval) = plot_traits
out_log10pval = -log10(out_pval)


library(viridis)
library(gplots)
pdf(file = outfile_heatmap,     # The directory you want to save the file in
        width = 10, # The width of the plot in inches
        height = 35)
par(mar=c(7,4,4,2)+0.1) 
out_map=heatmap.2(as.matrix(out_log10pval), dendrogram='row', col=viridis,
                    key=TRUE, cexRow=1,cexCol=1,margins=c(12,15),trace="none",srtCol=45)
dev.off()

ordered_labels = rownames(out_log10pval)[out_map$rowInd]


out_log10pval = out_log10pval[ordered_labels,]

dt2 <- out_log10pval %>%
    rownames_to_column() %>%
    gather(colname, value, -rowname)
dt2$rowname = factor(dt2$rowname, levels = unique(dt2$rowname))

ntest = nrow(dt2)
dt2$stars <- cut(dt2$value, breaks=c(-Inf, -log10(0.05/ntest), -log10(0.01/ntest), -log10(0.001/ntest), Inf), label=c("", "*", "**", "***"))    # Create column of significance labels

#scale_fill_gradient(low = "white", high = "red")

p = ggplot(dt2, aes(x = colname, y = rowname, fill = value)) +
    geom_tile() + scale_fill_viridis()    + ggtitle('-log10 p-values', ) + theme(plot.title = element_text(hjust = 0.5, size=15, face='bold'))
fig = p + geom_text(aes(label=stars), color="white", size=5, vjust=0.8) + scale_x_discrete(guide = guide_axis(angle = 45))
fig
ggsave(plot = fig, width = 8, height = 25, dpi = 300, filename = outfile_significance)


# WRITE RESULTS

write.csv(out, outfile_full)

write.csv(t(out_pval), outfile_pval)


# forest.data<-data.frame(label=traits,mean=hr,lower=lower.025,upper=upper.975, pval=pval)
# forest.data = forest.data[order(pval),]
# 
# forest.data$label <- factor(forest.data$label, levels=rev(forest.data$label))
# 
# library(ggplot2)
# fp <- ggplot(data=forest.data, aes(x=label, y=mean, ymin=lower, ymax=upper)) +
#     geom_pointrange() + 
#     geom_hline(yintercept=1, lty=2) +    # add a dotted line at x=1 after flip
#     coord_flip() +    # flip coordinates (puts labels on y axis)
#     xlab("Label") + ylab("Mean (95% CI)") +
#     theme_bw() # use a white background
# 
# ggsave(plot = fp, width = 8, height = 18, dpi = 300, filename = "/HDD/data/multitrait/cox/cox_diabetic_ventile5_full.pdf")
# print(fp)

# df = df[!is.na(df$FD_all),]
# rn = RankNorm(df$FD_all)
# df$rn = rn
# df$fd_2sd = as.factor(ifelse(rn<-2, 1, 0))
# 
# ggadjustedcurves(cox_model, variable='high_vein_df', data=df) + ylim(1,0.9) + grids(linetype = "dashed")
# 
# 
# # testing if residuals are decorrelated from time (Cox assumption)
# test.res = cox.zph(cox_model)
# ggcoxzph(test.res)
# 
# # testing influence of individual observations, by deleting them successively and seeing how that affects the prediction
# ggcoxdiagnostics(cox_model, type = 'dfbeta', linear.predictions = FALSE, ggtheme = theme_bw())

    # ggsurvplot(
    # fit = survfit(Surv(time=date_of_death_1, event=dead) ~ df_2sd, data = df), 
    # xlab = "Years", 
    # ylab = "Overall survival probability",
    # ylim(0.95,1))


# thres = mean(df_z$slope, na.rm=T) - 2*sd(df_z$slope, na.rm=T)
# df_z$low_fd = ifelse(df_z$slope<thres, 'FD < -2SD', 'FD > -2SD')
# cox_model = coxph( Surv(time=years_to_event, event=event) ~ low_fd + age + age2 + sex + cov1 + cov2 + cov3 + cov4 + cov5 + cov6 + cov7 + cov8 + cov9 + cov10 + cov11 + cov12 + cov13 + cov14 + cov15 + cov16 + cov17 + cov18 + cov19 + cov20, data=df_z)
# ggadjustedcurves(cox_model, variable='low_fd', data=df_z) + ylim(0.9,1) + xlim(0,11.0) + grids(linetype = "dashed") + scale_shape_discrete(labels = c("Female", "Male"))


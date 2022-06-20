library(survival)
library(survminer)
library(lubridate)

# df = read.csv('/SSD/home/michael/cox/cox_df_mortality.csv', na.strings=c("","NA"))
# df = read.csv('/SSD/home/michael/cox/cox_df_mortality_withSBP.csv', na.strings=c("","NA"))
# df = read.csv('/SSD/home/michael/cox/cox_df_mortality_ventile12.csv', na.strings=c("","NA"))
# df = read.csv('/SSD/home/michael/cox/cox_df_mortality_VD_Zekavat.csv', na.strings=c("","NA"))
# df = read.csv('/SSD/home/michael/cox/cox_df_mortality_noQC.csv', na.strings=c("","NA"))
df = read.csv('/SSD/home/michael/cox/cox_df_mortality_ventile2qc.csv', na.strings=c("","NA"))
# df = read.csv('/SSD/home/michael/cox/cox_df_mortality_no_qc.csv', na.strings=c("","NA"))


 # df = read.csv('/SSD/home/michael/cox/cox_vd_left_eye.csv', na.strings=c("","NA"))

df$sex = as.factor(df$sex)
df$age_2_at_visit = df$age_at_visit^2
df$date_of_visit = as.Date(df$date_of_visit, "%Y-%m-%d")
df$date_of_visit = year(df$date_of_visit) + (month(df$date_of_visit) - 1) / 12
# df$date_of_visit = as.numeric(df$date_of_visit)
df$date_of_death_0 = as.Date(df$date_of_death_0, "%Y-%m-%d")
df$date_of_death_0 = year(df$date_of_death_0) + (month(df$date_of_death_0) - 1) / 12
# df$date_of_death_1 = as.numeric(df$date_of_death_1)

# cox don't seem to support different start dates, so I set all to 0
df$date_of_death_0 = df$date_of_death_0 - df$date_of_visit
# min_date = min(df$date_of_visit)
# df$date_of_visit = df$date_of_visit - min_date
# df$date_of_death_1 = df$date_of_death_1 - min_date

end_time = max(df$date_of_death_0, na.rm=T) + 0.01# define end date of study for that time
df$date_of_death_0[is.na(df$date_of_death_0)] = end_time
hist(df$date_of_death_0)

# removing folks who joined later (instance 2 (or 1 in ukbb format)), improves HR, despite lower sample size
df = df[df$date_of_visit < 2012, ]
# but not randomly subsampling to a similar size
# df = sample_n(df, 57500)



# some died before visiting center
# cond = df$date_of_visit < df$date_of_death_1
# cond = df$date_of_death_1 >= 0
# cond[is.na(cond)] = T
# df = df[cond,]


dead = c()
for (i in df$date_of_death_0) {
  if ( i == end_time ) {
    dead = append(dead,0)
  }
  else {
    dead = append(dead,1)
  }
}

diabetic = c()
for (i in df$age_diabetes) {
  if ( is.na(i) ) {
    diabetic = append(diabetic,0)
  }
  else {
    diabetic = append(diabetic,1)
  }
}
diabetic = as.factor(diabetic)
df$diabetic = diabetic

stroke = c()
for (i in df$age_stroke) {
  if ( is.na(i) ) {
    stroke = append(stroke,0)
  }
  else {
    stroke = append(stroke,1)
  }
}
stroke = as.factor(stroke)
df$stroke = stroke

fd_all_threshold = mean(df$FD_all, na.rm=T) - 2*sd(df$FD_all, na.rm=T)
upper_thres = fd_all_threshold = mean(df$FD_all, na.rm=T) + 2*sd(df$FD_all, na.rm=T)
tmp = as.factor(ifelse(df$FD_all>upper_thres, 2, 0))
df$fd_2sd = as.factor(ifelse(df$FD_all<fd_all_threshold, 1, 0))
# df$fd_2sd[tmp==2]=2
df_all_threshold = mean(df$DF_all, na.rm=T) + 2*sd(df$DF_all, na.rm=T)
df$df_2sd = ifelse(df$DF_all>df_all_threshold, 1, 0)


thres = mean(df$VD_orig_all_Zekavat, na.rm=T) - 2*sd(df$VD_orig_all_Zekavat, na.rm=T)
df$vd_zeka = as.factor(ifelse(df$VD_orig_all_Zekavat<thres, 1, 0))
thres = mean(df$VD_orig_all_ours, na.rm=T) - 2*sd(df$VD_orig_all_ours, na.rm=T)
df$vd_our = as.factor(ifelse(df$VD_orig_all_ours<thres, 1, 0))


thres = mean(df$sbp_0, na.rm=T) + 2*sd(df$sbp_0, na.rm=T)
df$high_sbp = as.factor(ifelse(df$sbp_0>thres, 1, 0))
thres = mean(df$DF_artery, na.rm=T) - 0.5*sd(df$DF_artery, na.rm=T)
df$low_art_df = ifelse(df$DF_artery<thres, 1, 0)
thres = mean(df$DF_vein, na.rm=T) + 2*sd(df$DF_vein, na.rm=T)
df$high_vein_df = ifelse(df$DF_vein>thres, 1, 0)


df$hypertense = ifelse(df$sbp_0>140, 1, 0) # very simplified
tmp = df$fd_2sd
tmp[is.na(tmp)] = 0
df$fd_2sd_not_hypertense_not_diabetic = as.factor((as.numeric(tmp)-1) * (1 - as.numeric(df$hypertense)) * (2 - as.numeric(df$diabetic)))
df$fd_2sd_hypertense_diabetic = (as.numeric(tmp)-1) * (as.numeric(df$hypertense)) * (as.numeric(df$diabetic) - 1)
df$not_fd_2sd_hypertense_diabetic = (2 - as.numeric(tmp)) * (as.numeric(df$hypertense)) * (as.numeric(df$diabetic) - 1)

df$not_fd_2sd_hypertense_diabetic[is.na(df$not_fd_2sd_hypertense_diabetic)] = 0
df$fd_2sd_hypertense_diabetic[is.na(df$fd_2sd_hypertense_diabetic)] = 0
df$fd_2sd_not_hypertense_not_diabetic[is.na(df$fd_2sd_not_hypertense_not_diabetic)] = 0

df$pred_repl = "neither"
df$pred_repl[df$not_fd_2sd_hypertense_diabetic == 1] = "hypertense & diabetic"
df$pred_repl[df$fd_2sd_not_hypertense_not_diabetic == 1] = "low FD, not nypertense, not diabetic"
df$pred_repl[df$fd_2sd_hypertense_diabetic == 1] = "low FD, hypertense, diabetic"
df$pred_repl = as.factor(df$pred_repl)
df$pred_repl <- factor(df$pred_repl, levels = c("neither", "hypertense & diabetic", "low FD, not nypertense, not diabetic", "low FD, hypertense, diabetic"), ordered=T)

df = df[!is.na(df$FD_all),]
rn = RankNorm(df$FD_all)
df$fd_2sd = as.factor(ifelse(rn<-2, 1, 0))


cox_model = coxph( Surv(time=date_of_death_0, event=dead) ~ fd_2sd + age_at_visit + age_2_at_visit + sex + pc_1 + pc_2 + pc_3 + pc_4 + pc_5 + pc_6 + pc_7 + pc_8 + pc_9 + pc_10 + pc_11 + pc_12 + pc_13 + pc_14 + pc_15 + pc_16 + pc_17 + pc_18 + pc_19 + pc_20, data=df)
summary(cox_model)

ggadjustedcurves(cox_model, variable='high_vein_df', data=df) + ylim(1,0.9) + grids(linetype = "dashed")


# testing if residuals are decorrelated from time (Cox assumption)
test.res = cox.zph(cox_model)
ggcoxzph(test.res)

# testing influence of individual observations, by deleting them successively and seeing how that affects the prediction
ggcoxdiagnostics(cox_model, type = 'dfbeta', linear.predictions = FALSE, ggtheme = theme_bw())

  # ggsurvplot(
  # fit = survfit(Surv(time=date_of_death_1, event=dead) ~ df_2sd, data = df), 
  # xlab = "Years", 
  # ylab = "Overall survival probability",
  # ylim(0.95,1))

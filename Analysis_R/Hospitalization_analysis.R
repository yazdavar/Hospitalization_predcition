library (ggplot2)
library(caTools)
library(Amelia)
library(ggplot2)
library(VIM)
library(mice)
library(caret)
library(AppliedPredictiveModeling)
library(Hmisc)
library(RANN)
library(randomForest)
library(Boruta)
library(FSelector)
library(mlr)
library(corrplot)
library(missMDA)
library(tidyr)
library (dplyr)
library (broom)
library(DMwR)
library(corrplot)
library(PerformanceAnalytics)
library(heuristica)
library(irr)
library(lpSolve)
library(ggfortify)
library(ggpubr)
library("cowplot")
library(magrittr)
library("ggpubr")
require(reshape2)
require(scales)
#output_file = #'cdrn_psych_hosp_bef_18.csv'
#df <- read.csv("/Users/amir/code/Hospitalization_Weill_cornell/visit_freq_condition_type_time_series.tsv", header = TRUE,sep="\t", na.strings =c("")  )
df <- read.csv("/Users/amir/code/Hospitalization_Weill_cornell/visit_freq_condition_type_time_series_series_seasoned.tsv", header = TRUE,sep="\t", na.strings =c("")  )
names(df)
head(df)
df$visit_start_date <- as.Date(as.character(df$visit_start_date))


# Multiple line plot
ggplot(df, aes(x = visit_start_date, y = visit_frequency)) + 
  geom_line(aes(color = condition_source_value_type), size = 1) +
  scale_color_manual(values = c("#00AFBB", "#E7B800", "#00AFB0","#00DFBB")) +
  theme_minimal()


ggplot(df, aes(x=visit_start_date, y=visit_frequency, fill=condition_source_value_type)) + 
  geom_bar(stat="identity")


ggplot(df, aes(x = visit_start_date, y = visit_frequency)) + 
  geom_line(aes(color = condition_source_value_type), size = 1) +
  scale_color_manual(values = c("#00AFBB", "#E7B800", "#00AFB0","#00DFBB")) +
  theme_minimal()


#Healthcare Utlization visulization:

df_health_utlization <- read.csv("/Users/amir/code/Hospitalization_Weill_cornell/Healthcare_utilization_freq_time_sereies.tsv", header = TRUE,sep="\t", na.strings =c("")  )
names(df_health_utlization)
head(df_health_utlization)
df_health_utlization$visit_start_date <- as.Date(as.character(df_health_utlization$visit_start_date))


df_health_utlization <- filter(df_health_utlization, visit_concept_id_cat == 'Outpatient Visit' | visit_concept_id_cat ==  'Emergency Room Visit' | visit_concept_id_cat == "Inpatient Visit" |visit_concept_id_cat == "Ambulatory visit")

p <- ggplot(df_health_utlization, aes(x=visit_start_date, y=visit_occurrence_id, fill=visit_concept_id_cat)) + 
  geom_bar(stat="identity")


max <- as.Date("2017-01-01")
min <- as.Date("2012-01-01")
p <- p + scale_y_continuous(labels = comma) + scale_x_date(limits = c(min,max)) +
  theme_minimal()  
p + scale_x_date(date_labels = "%b/%Y")


#-----------------------------
#visualize the demographic information for the cdrn_psych_ file


cdrn_psych <- read.csv("/Users/amir/code/Hospitalization_Weill_cornell/cdrn_psych_sampled.tsv", header = TRUE,sep="\t", na.strings =c("")  )
names(cdrn_psych)
head(cdrn_psych)



#cdrn_psych$Psychiatric_Hospitalization <- as.factor(cdrn_psych$psych_hosp)
#cdrn_psych$sex <- as.factor(cdrn_psych$sex)

cdrn_psych$psych_hosp <- sapply(cdrn_psych$psych_hosp, function(x) if (x == 1 ) {"Case"} else {"Control"})
cdrn_psych$sex <- sapply(cdrn_psych$sex, function(x) if (x == 0 ) {"Female"} else {"Male"})

cdrn_psych <-rename(cdrn_psych, Gender = sex, Class = psych_hosp)
#df_reordered_remove_50S$Annotation <- sapply(df_reordered_remove_50S$Annotation, function(x) if (x == 'no' ) {"Control"} else {"Depressed"})
#cdrn_psych$Gender <- sapply(cdrn_psych$Gender, function(x) if (x == 1) {"1"} else {0})

#df_reordered$Human.Judge.for.Gender<-as.factor(df_reordered$Human.Judge.for.Gender)
tem_table <- table(cdrn_psych$Class, cdrn_psych$Gender)
tem_table
chisq <- chisq.test(tem_table)
chisq
corrplot(chisq$residuals, is.cor = FALSE)
dim(merged_gender)


cdrn_psych_cont <- table(cdrn_psych$Class, cdrn_psych$Gender) 
ftable(cdrn_psych_cont)
prop.table(cdrn_psych_cont)*100



mytable <- table(cdrn_psych$Class, cdrn_psych$Gender)
mytable
mytable_prop <- prop.table ( mytable,1)*100
chisq.test(mytable_prop)
ftable(mytable_prop)

#create PDF from non-depressed and depressed users
#yes_df_has_face_feature_removed_nan_with_age$age_text_cat_yes <-cut(yes_df_has_face_feature_removed_nan_with_age$Age, c(14,19,23,34,46,60))
#no_df_has_face_feature_removed_nan$age_text_cat_no <-cut(no_df_has_face_feature_removed_nan$Age, c(14,19,23,34,46,60))
#summary(yes_df_has_face_feature_removed_nan_with_age$age_text_cat)


#Levinson's Theory
cdrn_psych$age_cat <-cut(cdrn_psych$age, c(14,19,23,34,46,60,80))
summary(cdrn_psych$age_cat)

#plotting PDF

#density age
p <- ggplot(cdrn_psych, aes(x=age, colour=Class)) + geom_density()+ labs(x = "Age Distribution", y = "PDF" )  + guides(fill=guide_legend("my awesome title"))+ 
  theme_classic() 
p + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=23, face="bold"),
  axis.title.y = element_text(color="black", size=23, face="bold"),
  legend.text = element_text(size=20,  face="bold") ,
  legend.title = element_text(size=23,  face="bold"),
  axis.text.x = element_text(size = 15),
  axis.text.y = element_text(size = 15)
)                        #axis.title=element_text(size=14,face="bold"))


#density visit
p <- ggplot(cdrn_psych, aes(x=visit_count, colour=Class)) + geom_density()+ labs(x = "visit_count Distribution", y = "PDF" )  + guides(fill=guide_legend("my awesome title"))  + xlim(1, 200)+
  theme_classic() 
p + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=23, face="bold"),
  axis.title.y = element_text(color="black", size=23, face="bold"),
  legend.text = element_text(size=20,  face="bold") ,
  legend.title = element_text(size=23,  face="bold"),
  axis.text.x = element_text(size = 15),
  axis.text.y = element_text(size = 15)
)


#Box plot date difference gender
p_male <- ggplot(cdrn_psych, aes(x=Gender, y=date_diff, fill=Class)) + geom_boxplot()
p_male <- p_male + scale_y_continuous(limits = c(0, 3)) +labs(title = "", x = "", y = "Frequency of Male References")+theme_classic()
p_male <- p_male + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=23, face="bold"),
  axis.title.y = element_text(color="black", size=23, face="bold"),
  legend.text = element_text(size=25,  face="bold") ,
  legend.title = element_text(size=25,  face="bold"),
  axis.text.x = element_text(size = 15),
  axis.text.y = element_text(size = 15, face = "bold")
)
p_male



#Box plot visit gender
p_male <- ggplot(cdrn_psych, aes(x=Gender, y=visit_count, fill=Class)) + geom_boxplot() + ylim(1, 50)
p_male <- p_male + scale_y_continuous(limits = c(0, 3)) +labs(title = "", x = "", y = "Frequency of Male References")+theme_classic()
p_male <- p_male + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=23, face="bold"),
  axis.title.y = element_text(color="black", size=23, face="bold"),
  legend.text = element_text(size=25,  face="bold") ,
  legend.title = element_text(size=25,  face="bold"),
  axis.text.x = element_text(size = 15),
  axis.text.y = element_text(size = 15, face = "bold")
)
p_male



cdrn_psych_age <- ggplot(cdrn_psych, aes(x=age_cat, y=visit_count, fill=Class)) + stat_boxplot(geom="errorbar", width=.5,position = position_dodge(width = .75)) + geom_boxplot() + stat_summary(fun.y= mean, geom="point", shape=23, size=3 , position = position_dodge(width = .75)) +stat_summary(fun.y=mean, geom="smooth", linetype="dotdash", aes(color=paste("mean", Class),group=Class), lwd=0.65) +stat_summary(fun.y=mean, geom="smooth", linetype="F1", aes(color=paste("mean", Class),group=1), lwd=0.75) + ylim(1, 50)
#+ geom_jitter(position = position_jitter(0.1))
cdrn_psych_age
cdrn_psych_age <- cdrn_psych_age + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=23, face="bold"),
  axis.title.y = element_text(color="black", size=23, face="bold"),
  legend.text = element_text(size=25,  face="bold") ,
  legend.title = element_text(size=25,  face="bold"),
  axis.text.x = element_text(size = 15),
  axis.text.y = element_text(size = 15, face = "bold")
) 



#density date diff first and last visit
p <- ggplot(cdrn_psych, aes(x=date_diff, colour=Class)) + geom_density()+ labs(x = "date_diff Distribution", y = "PDF" )  + guides(fill=guide_legend("my awesome title"))  +
  theme_classic() 
p + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=23, face="bold"),
  axis.title.y = element_text(color="black", size=23, face="bold"),
  legend.text = element_text(size=20,  face="bold") ,
  legend.title = element_text(size=23,  face="bold"),
  axis.text.x = element_text(size = 15),
  axis.text.y = element_text(size = 15)
)


cdrn_psych_age <- ggplot(cdrn_psych, aes(x=age_cat, y=visit_count, fill=Class)) + stat_boxplot(geom="errorbar", width=.5,position = position_dodge(width = .75)) + geom_boxplot() + stat_summary(fun.y= mean, geom="point", shape=23, size=3 , position = position_dodge(width = .75)) +stat_summary(fun.y=mean, geom="smooth", linetype="dotdash", aes(color=paste("mean", Class),group=Class), lwd=0.65) +stat_summary(fun.y=mean, geom="smooth", linetype="F1", aes(color=paste("mean", Class),group=1), lwd=0.75) + ylim(1, 50)
#+ geom_jitter(position = position_jitter(0.1))
cdrn_psych_age
cdrn_psych_age <- cdrn_psych_age + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=23, face="bold"),
  axis.title.y = element_text(color="black", size=23, face="bold"),
  legend.text = element_text(size=25,  face="bold") ,
  legend.title = element_text(size=25,  face="bold"),
  axis.text.x = element_text(size = 15),
  axis.text.y = element_text(size = 15, face = "bold")
) 
cdrn_psych_age <- cdrn_psych_age  
cdrn_psych_age


cdrn_psych_case <- cdrn_psych[which(cdrn_psych$Class == "Case"),]
summary(cdrn_psych_case$date_diff)


      t_test <- t.test(yes_df$profile_imageFeatures.hueChannelVAR, no_df$profile_imageFeatures.hueChannelVAR)
t_test
p <- t_test$p.value
p
p <- p.adjust(p, method = "bonferroni")
p

if (p < 0.001){
  print ('***')
}else if (p < 0.01){
  print ('**')
} else if (p < 0.05){
  print ('*')
}



GrayScale <- ggplot(merged_image_numeric, aes(y=profile_imageFeatures.GrayScaleMean, x=Class)) + geom_boxplot()
GrayScale <- GrayScale + theme_classic()
GrayScale <- GrayScale + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=23, face="bold"),
  axis.title.y = element_text(color="black", size=23, face="bold"),
  legend.text = element_text(size=20,  face="bold") ,
  legend.title = element_text(size=23,  face="bold"),
  axis.text.x = element_text(size = 15),
  axis.text.y = element_text(size = 15, face= "bold")
) 
GrayScale <- GrayScale  + stat_summary(fun.y=mean, geom="point", shape=23, size=4)
GrayScale
#----------------------------------------
#Working on final files to decides what results to present in the paper

#-----------------------------
#visualize the demographic information for the cdrn_psych_ file


cdrn_final <- read.csv("/Users/amir/code/joe/OMOP_Hospital_Predict/Joe_results_server/Joe_results/cdrn_ccs_modified_label2.csv", header = TRUE,sep=",", na.strings =c("")  )
names(cdrn_final)
head(cdrn_final)

cdrn_final$psych_hosp <- sapply(cdrn_final$psych_hosp, function(x) if (x == 1 ) {"Case"} else {"Control"})
cdrn_final$sex <- sapply(cdrn_final$sex, function(x) if (x == 0 ) {"Female"} else {"Male"})

cdrn_final <-rename(cdrn_final, Gender = sex, Class = psych_hosp)
cdrn_final$Class<-as.factor(cdrn_final$Class)

#down smapling for doing statistical anlaysis
down_cdrn_final <- downSample(cdrn_final, cdrn_final$Class)
down_cdrn_final <- down_cdrn_final[, -c(303)]


down_cdrn_final$tuberculosis <- as.numeric(as.character(down_cdrn_final$tuberculosis))

#density age
p <- ggplot(down_cdrn_final, aes(x=tuberculosis, colour=Class)) + geom_density()+ labs(x = "Age Distribution", y = "PDF" )  + guides(fill=guide_legend("my awesome title"))+ 
  theme_classic() 
p + theme(
  plot.title = element_text(color="black", size=23, face="bold.italic"),
  axis.title.x = element_text(color="black", size=23, face="bold"),
  axis.title.y = element_text(color="black", size=23, face="bold"),
  legend.text = element_text(size=20,  face="bold") ,
  legend.title = element_text(size=23,  face="bold"),
  axis.text.x = element_text(size = 15),
  axis.text.y = element_text(size = 15)
)   

down_cdrn_final_class <- filter(down_cdrn_final, Class == "Case")
down_cdrn_final_Control <- filter(down_cdrn_final, Class == "Control")
#yes_df_sample <- sample_n(yes_df, 250)
#no_df_sample <- sample_n(no_df, 250)
t_test <- t.test(down_cdrn_final_class$enc, down_cdrn_final_Control$enc)

t_test
p <- t_test$p.value
p
p <- p.adjust(p, method = "bonferroni")
p

if (p < 0.0001){
  print ('***')
}else if (p < 0.01){
  print ('**')
} else if (p < 0.05){
  print ('*')
}


#-----------------Gender-------------
down_cdrn_final_class$Gender <- as.factor(down_cdrn_final_class$Gender)
down_cdrn_final_Control$Gender <- as.factor(down_cdrn_final_Control$Gender)

merged_gender<- rbind(down_cdrn_final_Control, down_cdrn_final_class)
merged_gender$Class <- as.factor(merged_gender$Class)

tem_table <- table(merged_gender$Class, droplevels(merged_gender$Gender))
tem_table
chisq <- chisq.test(tem_table)
chisq
corrplot(chisq$residuals, is.cor = FALSE)

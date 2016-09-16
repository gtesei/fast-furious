# load requied libraries --------------------------------------------------
library(data.table)


# load and transform people data ------------------------------------------
dfold = "C:/Users/gtesei/Desktop/Deloitte/C_Folder/Cognitive_Technologies/Machine_Learning/git/fast-furious/dataset/predicting-red-hat-business-value/"
ppl <- fread(paste0(dfold,"people.csv"))

### Recode logic to numeric
p_logi <- names(ppl)[which(sapply(ppl, is.logical))]

for (col in p_logi) {
  set(ppl, j = col, value = as.numeric(ppl[[col]]))
}
rm(p_logi)

### transform date
ppl[,date := as.Date(as.character(date), format = "%Y-%m-%d")]

# load activities ---------------------------------------------------------

# read and combine
activs <- fread(paste0(dfold,"act_train.csv"))
TestActivs <- fread(paste0(dfold,"act_test.csv"))

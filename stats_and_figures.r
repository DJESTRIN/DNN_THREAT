library(ggplot2)
path="/data/dje4001/DNN_THREAT/compiled_data.xlsx"
df<-read_exel(path)

p<-ggplot(data=df)

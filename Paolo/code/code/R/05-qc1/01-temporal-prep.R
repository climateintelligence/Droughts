# some figures for province meeting

library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(directlabels)
library(readxl)
library(writexl)


# prep data ---------------------------------------------------------------



dat_meta_1 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/meta_long_HN_HS.rds")
dat_1 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/data_long_HN_HS.rds")

dat_meta_2 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1787hn_1879hs-1960/r-data/meta_long_HN_HS.rds")
dat_2 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1787hn_1879hs-1960/r-data/data_long_HN_HS.rds")



# dat_1[, month := month(Date)]
# dat_2[, month := month(Date)]
# 
# dat_1[, season_year := year(Date)]
# dat_2[, season_year := year(Date)]
# 
# dat_1[month <= 8, season_year := season_year - 1L]
# dat_2[month <= 8, season_year := season_year - 1L]

dat_all <- rbind(dat_2, dat_1)
setkey(dat_all, Provider, Name, Date)


# temporal consistency HS -------------------------------------------------



dat_all[, HS_lag := shift(HS, type = "lag"), .(Provider, Name)]
dat_all[, HS_lead := shift(HS, type = "lead"), .(Provider, Name)]

dat_all[abs(HS - HS_lag) > 50 & abs(HS - HS_lead) > 50]




# temporal consistency plots ----------------------------------------------------


dat_check <- dat_all[abs(HS - HS_lag) > 50 & abs(HS - HS_lead) > 50 & 
                       sign(HS - HS_lag) != sign(HS_lead - HS)]
dat_check[, idn := 1:.N]
dat_check[, idn_grp9 := floor((idn-1) /9)]

dat_window <- dat_check[, 
                        .(Date = seq(Date - 30, Date + 30, by =  "day")),
                        .(Name, Date_error = Date, idn, idn_grp9)]

dat_plot <- merge(dat_window, dat_all, by = c("Name", "Date"))

pdf("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/v02/fig/temporal-consistency.pdf",
    width = 12, height = 8)

for(i_idn_grp9 in sort(unique(dat_check$idn_grp9))){
  dat_plot[idn == i_idn_grp9] %>% 
    .[1, paste0(Provider, " / ", Name, " / ", Date_error)] -> tit
  
  gg <- 
    dat_plot[idn_grp9 == i_idn_grp9] %>% 
    ggplot(aes(Date))+
    geom_point(aes(y = HS, colour = "HS"))+
    geom_point(aes(y = HN, colour = "HN"))+
    geom_vline(aes(xintercept = Date_error), linetype = "dashed")+
    facet_wrap(~idn + Name + Date_error, scales = "free")+
    theme_bw()+
    ylab("HS / HN  [cm]")
  
  
  print(gg)
  
  
}

dev.off()


# tables ------------------------------------------------------------------

dat_window_table <- dat_check[, 
                        .(Date = seq(Date - 2, Date + 2, by =  "day")),
                        .(Name, Date_error = Date, idn, idn_grp9)]

dat_table <- merge(dat_window_table, dat_all, by = c("Name", "Date"))

setkey(dat_table, idn, Date)

dat_out <- dat_table[, .(idn, Name, Date_error, Date, dd = Date == Date_error, HN, HS)]
dat_out[, dd := ifelse(dd, "X", "")]

write_xlsx(
  dat_out,
  "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/v02/table/temporal-consistency_empty.xlsx"
)




# pre-filled from v01 -----------------------------------------------------

dat_tc <- read_excel("manual-qc/v01/temporal-consistency_filled.xlsx")
setDT(dat_tc)
dat_tc %>% str
dat_tc[, Date_error := as.Date(Date_error)]
dat_tc[, Date := as.Date(Date)]
setkey(dat_tc, Name, Date, Date_error)

dat_out_prefilled <- copy(dat_out)
setkey(dat_out_prefilled, Name, Date, Date_error)

dat_out_prefilled %>% 
  merge(dat_tc[, .(Name, Date_error, Date, HS_error, HN_error, OK)],
        all.x = T) -> dat_out_prefilled2


setkey(dat_out_prefilled2, idn, Date)
setcolorder(dat_out_prefilled2, "idn")

write_xlsx(
  dat_out_prefilled2,
  "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/v02/table/temporal-consistency_prefilled-v01.xlsx"
)



# EOF ---------------------------------------------------------------------



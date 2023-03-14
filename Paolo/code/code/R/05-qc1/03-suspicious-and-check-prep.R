# prep tables and figs for zero == NA issue

library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(directlabels)
library(readxl)
library(writexl)
library(foreach)
library(fs)


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

dat_meta <- unique(rbind(dat_meta_1, dat_meta_2))

dat_all[, month := month(Date)]


dat_all[, HN := as.integer(round(HN))]
dat_all[, HS := as.integer(round(HS))]



# read in stns ------------------------------------------------------------

fread("manual-qc/v02/series-to-check.txt")
fread("manual-qc/v02/series-to-check_from-zeroNA.txt")

dat_meta[grepl("edf", Name)]
dat_meta[grepl("glac", Name)]




stns_to_check <- c(
  fread("manual-qc/v02/series-to-check.txt") %>% .$Name,
  fread("manual-qc/v02/series-to-check_from-zeroNA.txt") %>% .$Name
)

stns_to_check %>% unique %>% sort -> stns_to_check

stopifnot(all(stns_to_check %in% dat_meta$Name))

# stns_to_check[!stns_to_check %in% dat_meta$Name]
# stns_to_check[!stns_to_check %in% dat_all$Name]
# dat_meta[startsWith(Name, "St_")]

# for manual check: plots and tables ----------------------------------------------------

for(i_stn in stns_to_check){
  
  dat_i_stn <- dat_all[Name == i_stn]
  dat_i_stn[, year := year(Date)]
  
  dat_i_stn[, idn := year - min(year)]
  dat_i_stn[, idn_grp := floor(idn / 2)]
  
  fn_out <- paste0("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/v02/fig/series-to-check/", i_stn, ".pdf")
  if(file_exists(fn_out)) next
  
  pdf(fn_out,
      width = 14, height = 7)
  
  for(i_idn_grp in sort(unique(dat_i_stn$idn_grp))){
    
    gg <- 
      dat_i_stn[idn_grp == i_idn_grp] %>% 
      ggplot(aes(Date))+
      geom_point(aes(y = HS, colour = "HS"))+
      geom_point(aes(y = HN, colour = "HN"), size = 0.5)+
      facet_wrap(~year, scales = "free", nrow = 2)+
      scale_x_date(date_labels = "%b")+
      theme_bw()+
      ylab("HS / HN  [cm]")
    
    
    print(gg)
    
    
  }
  
  dev.off()
  
  
  # table
  dat_table <- dat_i_stn[, .(year, Date, HN, HS, to_remove = NA_integer_)]
  
  setkey(dat_table, Date)
  
  write_xlsx(
    dat_table,
    paste0("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/v02/table/series-to-check/", i_stn, ".xlsx")
  )
  
}

# overview table

dat_out <- data.table(Name = stns_to_check,
                      OK = NA,
                      remove_some_values = NA,
                      remove_whole_series = NA)

dat_out %>% 
  write_xlsx("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/v02/table/series-to-check-overview_empty.xlsx")




# prefill overview table and copy files -----------------------------------

dat_out_v01 <- read_excel("manual-qc/v01/suspicious-and-check-overview_filled.xlsx")
setDT(dat_out_v01)

dat_out_prefilled <- copy(dat_out)

dat_out_prefilled[, .(Name)] %>% 
  merge(dat_out_v01, all.x = T) -> dat_out_prefilled2


dat_out_prefilled2 %>% 
  write_xlsx("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/v02/table/series-to-check-overview_prefilled-v01.xlsx")


# copy prefilled files and empty ones, too
library(fs)
files_prefilled_in <- path("manual-qc/v01/suspicious-and-check_filled/", 
                           dat_out_prefilled2[remove_some_values == 1, Name], ext = "xlsx")

files_prefilled_out <- path("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/v02/table/series-to-check_prefilled-v01/",
                            path_file(files_prefilled_in))

file_copy(files_prefilled_in, files_prefilled_out, overwrite = T)



files_copy_in <- path("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/v02/table/series-to-check/", 
                      dat_out_prefilled2[is.na(remove_some_values) | remove_some_values != 1, Name], 
                      ext = "xlsx")

files_copy_out <- path("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/v02/table/series-to-check_prefilled-v01/",
                       path_file(files_copy_in))

file_copy(files_copy_in, files_copy_out, overwrite = T)








# EOF ---------------------------------------------------------------------



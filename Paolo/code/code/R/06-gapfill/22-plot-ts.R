# plot all 1961-2020 series
# with gapfill id


library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(directlabels)
library(foreach)
library(fs)


dat_hshn <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1961-2020/data_long_HN_HS.rds")
dat_hs_gapf <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/gapfill-01-all.rds") 

dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1961-2020/meta_long_HN_HS.rds")

stns_to_check <- unique(dat_hs_gapf$Name)

cols <- setNames(scales::hue_pal()(3),
                 c("1_HS_original", "2_HS_filled", "3_HN"))

for(i_stn in stns_to_check){
  
  dat_i_stn <- merge(dat_hs_gapf[Name == i_stn],
                     dat_hshn[Name == i_stn, .(Name, Date, HN)],
                     by = c("Name", "Date"),
                     all.x = T)
  dat_i_stn[, year := year(Date)]
  
  dat_i_stn[, idn := year - min(year)]
  dat_i_stn[, idn_grp := floor(idn / 2)]
  
  dat_i_stn[, HS_fillcode_fct := fct_recode(factor(HS_fillcode),
                                            "1_HS_original" = "1",
                                            "2_HS_filled" = "222")]
  
  i_prov <- dat_meta[Name == i_stn, Provider]
  
  fn_out <- paste0("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/fig/timeseries/",
                   i_prov, "/", i_stn, ".pdf")
  dir_create(path_dir(fn_out))
  if(file_exists(fn_out)) next
  
  pdf(fn_out,
      width = 14, height = 7)
  
  for(i_idn_grp in sort(unique(dat_i_stn$idn_grp))){
    
    gg <- 
      dat_i_stn[idn_grp == i_idn_grp] %>% 
      ggplot(aes(Date))+
      geom_point(aes(y = HS, colour = HS_fillcode_fct))+
      geom_point(aes(y = HN, colour = "3_HN"), size = 0.5)+
      facet_wrap(~year, scales = "free", nrow = 2)+
      scale_x_date(date_labels = "%b")+
      scale_color_manual(values = cols)+
      theme_bw()+
      ylab("HS / HN  [cm]")
    
    
    print(gg)
    
    
  }
  
  dev.off()
  
  
  
}
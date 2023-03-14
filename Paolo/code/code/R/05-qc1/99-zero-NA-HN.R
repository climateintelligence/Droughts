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

dat_meta <- unique(rbind(dat_meta_1, dat_meta_2))

dat_all[, month := month(Date)]

dat_all[, HN := as.integer(round(HN))]
dat_all[, HS := as.integer(round(HS))]

# moving window  ----------------------------------------------------------


dat_values <- dat_all[!is.na(HN) & month %in% c(12, 1:2) &
                      Name != "La_Thuile_Ospizio_Piccolo_San_Bernardo",
                    .(n_avail = .N,
                      mean_hn = mean(HN),
                      sum_hn = sum(HN) / length(unique(year(Date))),
                      n_zero = sum(HN == 0),
                      mean_non_zero = mean(HN[HN != 0])),
                    .(Name)]
dat_values[, frac_zero := n_zero / n_avail]

dat_values %>% 
  merge(dat_meta, by = "Name") %>% 
  ggplot(aes(Elevation, mean_hn, colour = frac_zero))+
  geom_point()+
  scale_color_binned(type = "viridis", n.breaks = 10)

dat_values %>% 
  merge(dat_meta, by = "Name") %>% 
  ggplot(aes(Elevation, sum_hn, colour = frac_zero))+
  geom_point()+
  scale_color_binned(type = "viridis", n.breaks = 10)



# not so sure if worth it?!


dat_loop <- merge(dat_values[, .(Name)], 
                  dat_meta[, .(Name, Elevation)])
dat_loop[, delta_alt := 100]
dat_loop[Elevation < 300, delta_alt := 200]
dat_loop[Elevation > 1000, delta_alt := 200]
dat_loop[Elevation > 2000, delta_alt := 300]
dat_loop[Elevation > 2500, delta_alt := 500]

dat_loop$Elevation %>% qplot(binwidth = 50)

dat_loop_out <- foreach(
  i = 1:nrow(dat_loop),
  .final = rbindlist
) %do% {
  
  i_stn <- dat_loop[i, Name]
  elev_range <- dat_loop[i, Elevation + delta_alt * c(-1, 1)]
  stns_range <- dat_loop[Elevation %between% elev_range, Name]
  
  xx_frac_zero <-  dat_values[Name %chin% stns_range, frac_zero]
  xx_mean_hs <-  dat_values[Name %chin% stns_range, mean_hs]
  
  data.table(
    dat_loop[i],
    frac_zero_q05 = quantile(xx_frac_zero, 0.05),
    frac_zero_q95 = quantile(xx_frac_zero, 0.95),
    mean_hs_q05 = quantile(xx_mean_hs, 0.05),
    mean_hs_q95 = quantile(xx_mean_hs, 0.95),
    
    frac_zero_mean = mean(xx_frac_zero),
    frac_zero_sd = sd(xx_frac_zero),
    mean_hs_mean = mean(xx_mean_hs),
    mean_hs_sd = sd(xx_mean_hs)
  )
  
  
}


# both bounds --------------------------------------------------------



dat_plot <- merge(dat_values, dat_loop_out, by = c("Name"))
dat_plot[, in_zero_qr := frac_zero %between% list(frac_zero_q05, frac_zero_q95)]
dat_plot[, in_hs_qr := mean_hs %between% list(mean_hs_q05, mean_hs_q95)]
dat_plot[, in_zero_sd := (frac_zero - frac_zero_mean) %between% 
           list(-1.96 * frac_zero_sd, +1.96 * frac_zero_sd)]
dat_plot[, in_hs_sd := (mean_hs - mean_hs_mean) %between% 
           list(-1.96 * mean_hs_sd, +1.96 * mean_hs_sd)]

dat_plot %>% with(table(in_zero_qr, in_zero_sd))
dat_plot %>% with(table(in_hs_qr, in_hs_sd))


dat_plot %>% 
  ggplot(aes(Elevation, mean_hs, colour = frac_zero))+
  geom_point()+
  scale_color_binned(type = "viridis", n.breaks = 10)+
  facet_grid(in_zero_qr ~ in_zero_sd, labeller = label_both)

dat_plot %>% 
  ggplot(aes(Elevation, mean_hs, colour = frac_zero))+
  geom_point()+
  scale_color_binned(type = "viridis", n.breaks = 10)+
  facet_grid(in_hs_qr ~ in_hs_sd, labeller = label_both)



# one bound --------------------------------------------------------

dat_plot[, in_zero_qr := frac_zero %between% list(-Inf, frac_zero_q95)]
dat_plot[, in_hs_qr := mean_hs %between% list(mean_hs_q05, Inf)]
dat_plot[, in_zero_sd := (frac_zero - frac_zero_mean) %between% 
           list(-1.96 * Inf, +1.96 * frac_zero_sd)]
dat_plot[, in_hs_sd := (mean_hs - mean_hs_mean) %between% 
           list(-1.96 * mean_hs_sd, +1.96 * Inf)]

dat_plot %>% with(table(in_zero_qr, in_zero_sd))
dat_plot %>% with(table(in_hs_qr, in_hs_sd))


dat_plot %>% 
  ggplot(aes(Elevation, mean_hs, colour = frac_zero))+
  geom_point()+
  scale_color_binned(type = "viridis", n.breaks = 10)+
  facet_grid(in_zero_qr ~ in_zero_sd, labeller = label_both)

dat_plot %>% 
  ggplot(aes(Elevation, mean_hs, colour = frac_zero))+
  geom_point()+
  scale_color_binned(type = "viridis", n.breaks = 10)+
  facet_grid(in_hs_qr ~ in_hs_sd, labeller = label_both)


dat_plot %>% with(table(in_hs_qr, in_zero_qr))

dat_plot %>% 
  ggplot(aes(Elevation, mean_hs, colour = frac_zero))+
  geom_point()+
  scale_color_binned(type = "viridis", n.breaks = 10)+
  facet_grid(in_hs_qr ~ in_zero_qr, labeller = label_both)






# -> most sense:
#    - quantile range 
#    - one bound
#    - hs or zero outside (not AND)


stns_to_check <- dat_plot[in_zero_qr == FALSE | in_hs_qr == FALSE, Name]




# for manual check: plots and tables ----------------------------------------------------

for(i_stn in stns_to_check){
  
  dat_i_stn <- dat_all[Name == i_stn]
  dat_i_stn[, year := year(Date)]
  
  dat_i_stn[, idn := year - min(year)]
  dat_i_stn[, idn_grp := floor(idn / 2)]
  
  
  pdf(paste0("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/fig/zero-NA/", i_stn, ".pdf"),
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
  dat_table <- dat_i_stn[, .(year, Date, HN, HS, zeroNA = NA_integer_)]
  
  setkey(dat_table, Date)
  
  write_xlsx(
    dat_table,
    paste0("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/table/zero-NA/", i_stn, ".xlsx")
  )
  
}

# overview table

data.table(Name = stns_to_check,
           OK = NA,
           zero_NA = NA) %>% 
  write_xlsx("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/table/zero-NA-overview_empty.xlsx")



# EOF ---------------------------------------------------------------------



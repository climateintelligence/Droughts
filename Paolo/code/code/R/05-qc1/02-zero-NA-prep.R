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


# zero NA -------------------------------------------------

dat_summ <- dat_all[!is.na(HS) & month %in% c(12, 1:2),
                    .(n_avail = .N,
                      mean = mean(HS),
                      n_zero = sum(HS == 0),
                      mean_non_zero = mean(HS[HS != 0]),
                      n_low = sum(HS < 10),
                      mean_non_low = mean(HS[HS > 10])),
                    .(Name)]

with(dat_summ[n_zero != 0], plot(n_zero/n_avail, mean_non_zero))
with(dat_summ[n_zero != 0 & n_zero/n_avail > 0.2], plot(n_zero/n_avail, mean_non_zero))
with(dat_summ[n_zero != 0], plot(n_zero/n_avail, mean/mean_non_zero))
with(dat_summ[n_low != 0], plot(n_low/n_avail, mean_non_low))
dat_summ[n_zero == 0]
with(dat_summ[n_zero != 0], plot(mean, mean_non_zero))
with(dat_summ[n_low != 0], plot(mean, mean_non_low))



dat_summ2 <- merge(dat_summ, dat_meta[, .(Name, Elevation)])


dat_summ2 %>% 
  ggplot(aes(Elevation, mean, colour = n_zero/n_avail))+
  geom_point()+
  scale_color_binned(type = "viridis", n.breaks = 10)

# check the outlier
# dat_summ2[mean > 500]
# dat_zz <- dat_all[Name == "La_Thuile_Ospizio_Piccolo_San_Bernardo"]
# dat_zz[!is.na(HS)]

dat_summ2[Elevation > 1000] %>% 
  ggplot(aes(Elevation, mean_non_zero, 
             colour = n_zero/n_avail, alpha = n_zero/n_avail > 0.2))+
  geom_point()+
  scale_color_binned(type = "viridis", n.breaks = 10)

dat_summ2[Elevation > 1000 & n_zero/n_avail > 0.2]





dat_summ_month <- dat_all[!is.na(HS) & month %in% c(12, 1:2) &
                            Name != "La_Thuile_Ospizio_Piccolo_San_Bernardo",
                          .(n_avail = .N,
                            mean = mean(HS),
                            n_zero = sum(HS == 0),
                            mean_non_zero = mean(HS[HS != 0]),
                            n_low = sum(HS < 10),
                            mean_non_low = mean(HS[HS > 10])),
                          .(Name, month)]



dat_summ_month2 <- merge(dat_summ_month, dat_meta[, .(Name, Elevation)])

dat_summ_month2 %>% 
  ggplot(aes(Elevation, mean, colour = n_zero/n_avail))+
  geom_point()+
  scale_color_binned(type = "viridis", n.breaks = 10)+
  facet_wrap(~month, scales = "free_y")



# try a moving window detection -------------------------------------------


dat_values <- dat_all[!is.na(HS) & month %in% c(12, 1:2),
                    .(n_avail = .N,
                      mean_hs = mean(HS),
                      n_zero = sum(HS == 0),
                      mean_non_zero = mean(HS[HS != 0])),
                    .(Name)]
dat_values[, frac_zero := n_zero / n_avail]


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

dat_plot %>% with(table(in_hs_qr, in_zero_qr))

dat_plot %>% 
  ggplot(aes(Elevation, mean_hs, colour = frac_zero))+
  geom_point()+
  scale_color_binned(type = "viridis", n.breaks = 10)+
  facet_grid(in_hs_qr ~ in_zero_qr, labeller = label_both)

# one bound --------------------------------------------------------

dat_plot <- merge(dat_values, dat_loop_out, by = c("Name"))
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


# plot for presentation
dat_plot %>% 
  ggplot(aes(Elevation, mean_hs, colour = frac_zero))+
  geom_point()+
  scale_color_binned(type = "viridis", n.breaks = 10)+
  theme_bw()+
  ylab("Mean DJF HS [cm]")+xlab("Elevation [m]")
ggsave(width = 8, height = 4,
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/03_july_meeting/fig/zeroNA-hs by elev_v02.png")


# -> most sense:
#    - quantile range 
#    - one bound
#    - hs or zero outside (not AND)


stns_to_check <- dat_plot[in_zero_qr == FALSE | in_hs_qr == FALSE, Name]

# suspicious
dat_plot[Elevation < 1000 & mean_hs > 70]
dat_plot[Elevation > 1000 & Elevation < 1300 & mean_hs > 80]
dat_plot[Elevation > 1000 & Elevation < 1600 & mean_hs > 95]
dat_plot[Elevation > 1500 & Elevation < 2000 & mean_hs > 150]
dat_plot[Elevation > 2000 & mean_hs > 200]
dat_plot[Elevation > 3000 & mean_hs < 100]

rbindlist(list(
  dat_plot[Elevation < 1000 & mean_hs > 70],
  dat_plot[Elevation > 1000 & Elevation < 1300 & mean_hs > 80],
  dat_plot[Elevation > 1000 & Elevation < 1600 & mean_hs > 95],
  dat_plot[Elevation > 1500 & Elevation < 2000 & mean_hs > 150],
  dat_plot[Elevation > 2000 & mean_hs > 200],
  dat_plot[Elevation > 3000 & mean_hs < 100]
)) %>% .[, .(Name = sort(unique(Name)), reason = "high HS relative to alt")] %>% 
  fwrite("manual-qc/v02/series-to-check_from-zeroNA.txt")

# for manual check: plots and tables ----------------------------------------------------

for(i_stn in stns_to_check){
  
  dat_i_stn <- dat_all[Name == i_stn]
  dat_i_stn[, year := year(Date)]
  
  dat_i_stn[, idn := year - min(year)]
  dat_i_stn[, idn_grp := floor(idn / 2)]
  
  
  pdf(paste0("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/v02/fig/zero-NA/", i_stn, ".pdf"),
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
    paste0("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/v02/table/zero-NA/", i_stn, ".xlsx")
  )
  
}

# overview table

dat_out_overview <- data.table(Name = stns_to_check,
                               OK = NA,
                               zero_NA = NA)

dat_out_overview %>% 
  write_xlsx("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/v02/table/zero-NA-overview_empty.xlsx")




# prefill overview table and copy files -----------------------------------

dat_0na_overview <- read_excel("manual-qc/v01/zero-NA-overview_filled.xlsx")
setDT(dat_0na_overview)

dat_out_overview_prefilled <- copy(dat_out_overview)

dat_out_overview_prefilled[, .(Name)] %>% 
  merge(dat_0na_overview, all.x = T) -> dat_out_overview_prefilled2


dat_out_overview_prefilled2 %>% 
  write_xlsx("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/v02/table/zero-NA-overview_prefilled-v01.xlsx")


# copy prefilled files and empty ones, too
library(fs)
files_prefilled_in <- path("manual-qc/v01/zero-NA_filled/", 
                           dat_out_overview_prefilled2[zero_NA == 1, Name], ext = "xlsx")

files_prefilled_out <- path("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/v02/table/zero-NA_prefilled-v01/",
                            path_file(files_prefilled_in))

file_copy(files_prefilled_in, files_prefilled_out)



files_copy_in <- path("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/v02/table/zero-NA/", 
                      dat_out_overview_prefilled2[is.na(zero_NA) | zero_NA != 1, Name], ext = "xlsx")

files_copy_out <- path("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/v02/table/zero-NA_prefilled-v01/",
                       path_file(files_copy_in))

file_copy(files_copy_in, files_copy_out)



# EOF ---------------------------------------------------------------------



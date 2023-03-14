# prep data for manual check
# spatial consistency


library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(directlabels)
library(readxl)
library(writexl)
library(fs)

dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/07_SPATIAL_CONSISTENCY/data-to-check/meta_wide_HS.rds")
dat_hs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/07_SPATIAL_CONSISTENCY/data-to-check/data_long_HS.rds")
dat_hs2 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/data_long_HS.rds")
dat_hs <- mitmatmisc::add_month_fct(dat_hs, 10)




# determine stns to check -------------------------------------------------

dat_sim <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/07_SPATIAL_CONSISTENCY/SC_5REF/MonthlySim_5Ref.rds")
setDT(dat_sim)

dat_obs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/07_SPATIAL_CONSISTENCY/data-to-check/data_wide_HS.rds")
dat_obs_sim <- merge(
  dat_obs %>% melt(id.vars = c("year", "month"), value.name = "obs", variable.name = "Name"),
  dat_sim %>% melt(id.vars = c("year", "month"), value.name = "sim", variable.name = "Name"),
  by = c("year", "month", "Name"),
  all = T
)


dat_djf <- dat_obs_sim[month %in% c(12, 1, 2), 
                       .(bias = mean(sim - obs, na.rm = T),
                         mae = mean(abs(sim - obs), na.rm = T),
                         rmse = sqrt(mean( (sim - obs)^2, na.rm = T )),
                         r2 = cor(sim, obs, use = "p")^2,
                         nn = sum(!is.na(obs) & !is.na(sim))),
                       .(Name)]

dat_metrics_djf <- merge(dat_djf, dat_meta, by = "Name")



dat_metrics_djf[, Ele_fct := cut(Elevation, 
                                 breaks = c(0, 250, 500, 750, 1000, 1250, 1500, 2000, 3000),
                                 dig.lab = 5)]

dat_metrics_djf_summ <- dat_metrics_djf[!is.na(bias), 
                                        .(bias_mu = mean(bias),
                                          bias_sd = sd(bias),
                                          bias_med = median(bias),
                                          bias_q25 = quantile(bias, 0.25),
                                          bias_q75 = quantile(bias, 0.75),
                                          bias_iqr = IQR(bias)),
                                        .(Ele_fct)]

dat_metrics_djf2 <- merge(dat_metrics_djf, dat_metrics_djf_summ, by = "Ele_fct")


dat_metrics_djf2[, in_sd := between(bias,
                                    bias_mu - 1.96 * bias_sd,
                                    bias_mu + 1.96 * bias_sd,)]

dat_metrics_djf2[, in_ab_mae := Elevation/1000*13 > mae]

stns_to_check <- dat_metrics_djf2[in_sd == F | r2 < 0.5 | in_ab_mae == F | is.na(bias),
                                  Name]




# overview table ----------------------------------------------------------


dat_table <- dat_meta[Name %in% stns_to_check,
                      .(Provider, Name, OK = NA, suspicious = NA)]

# dat_table <- data.table(Name = stns_to_check, OK = NA, suspicious = NA)

setcolorder(dat_table, c("Provider", "Name"))

write_xlsx(
  dat_table,
  "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/07_SPATIAL_CONSISTENCY/manual-check/spatcons_overview_table_empty.xlsx"
)


# plot monthly ------------------------------------------------------------
# 
# for(i_stn in stns_to_check){
#   
#   dat_plot <- dat_hs[Name == i_stn & month %in% c(11,12,1:5)]
#   
#   tit <- dat_meta[Name == i_stn, paste0(Name, ", ", Elevation, "m, ", Provider)]
#   
#   gg <- dat_plot %>% 
#     ggplot(aes(year, HS))+
#     geom_point(na.rm = T)+
#     geom_smooth(se = F, method = "lm", formula = y ~ x, colour = "grey20", na.rm = T)+
#     facet_wrap(~month_fct)+
#     # scale_color_binned(type = "viridis", n.breaks = 10, limits = c(0,1))+
#     theme_bw()+
#     # theme(legend.position = "bottom")+
#     ggtitle(tit)+
#     xlab(NULL)+ylab("Mean monthly HS [cm]")+
#     xlim(1961, 2020)
#   
#   fn_out <- paste0("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/07_SPATIAL_CONSISTENCY/manual-check/plots-monthly-ts/",
#                    dat_meta[Name == i_stn, Provider],
#                    "_",
#                    i_stn,
#                    ".png")
#   
#   ggsave(gg,
#          filename = fn_out,
#          width = 12, height = 6)
#   
# }


# copy plots from Alice ---------------------------------------------------



for(i_stn in stns_to_check){
  
  fn_base <- dat_meta[Name == i_stn, paste0(Provider, "_", Name, ".png")]
  
  fn_in <- path("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/07_SPATIAL_CONSISTENCY/SC_5REF/plots/",
                fn_base)
  fn_out <- path("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/07_SPATIAL_CONSISTENCY/manual-check/plots-alice/",
                 fn_base)
  
  if(file_exists(fn_in)) file_copy(fn_in, fn_out, overwrite = T)
  
}


# plot monthly obs vs sim DJF ---------------------------------------------



for(i_stn in stns_to_check){
  
  dat_plot <- dat_obs_sim[Name == i_stn & month %in% c(11, 12, 1:4)]
  dat_plot2 <- merge(dat_plot, 
                     dat_hs2[Name == i_stn & month %in% c(11, 12, 1:4)], 
                     all = T)
  
  mitmatmisc::add_month_fct(dat_plot2, 10)
  
  tit <- dat_meta[Name == i_stn, paste0(Name, ", ", Elevation, "m, ", Provider)]
  
  gg <-
  dat_plot2 %>% 
    ggplot(aes(year))+
    geom_point(aes(y = HS, colour = frac_gapfilled), na.rm = T)+
    geom_line(aes(y = HS), na.rm = T, alpha = 0.5)+
    geom_point(aes(y = sim), na.rm = T, colour = "red", alpha = 0.5)+
    geom_line(aes(y = sim), na.rm = T, colour = "red", alpha = 0.5)+
    # geom_smooth(se = F, method = "lm", formula = y ~ x, colour = "grey20", na.rm = T)+
    facet_wrap(~month_fct, scales = "free_y")+
    scale_color_binned(type = "viridis", n.breaks = 10, limits = c(0,1))+
    theme_bw()+
    # theme(legend.position = "bottom")+
    ggtitle(tit)+
    xlab(NULL)+ylab("Mean monthly HS [cm]")+
    xlim(1961, 2020)
  
  
  fn_out <- paste0("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/07_SPATIAL_CONSISTENCY/manual-check/plots-monthly-ts/",
                   dat_meta[Name == i_stn, Provider],
                   "_",
                   i_stn,
                   ".png")
  
  ggsave(gg,
         filename = fn_out,
         width = 12, height = 6)
  
}








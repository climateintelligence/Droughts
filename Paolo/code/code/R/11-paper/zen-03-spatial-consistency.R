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

# dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/07_SPATIAL_CONSISTENCY/data-to-check/meta_wide_HS.rds")
dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/07_SPATIAL_CONSISTENCY/data-to-check/backup/meta_wide_HS.rds")
# dat_hs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/07_SPATIAL_CONSISTENCY/data-to-check/data_long_HS.rds")
dat_hs2 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/data_long_HS.rds")
# dat_hs <- mitmatmisc::add_month_fct(dat_hs, 10)

all_providers <- sort(unique(dat_meta$Provider))


# copy alices plots, and rename -------------------------------------------
# 
# files_alice_full <- dir_ls("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/07_SPATIAL_CONSISTENCY/SC_5REF/plots/",
#                       regexp = "png$")
# 
# for(i_fn in files_alice_full){
#   
#   path_file(i_fn) %>% 
#     path_ext_remove() -> i_prov_stn
#   
#   i_prov_stn %>% 
#     stringr::str_locate(all_providers) -> prov_match
#   
#   i_match_prob <- max(which(!is.na(prov_match[, 1])))
#   
#   fn_out <- path("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/zenodo/aux_paper_spatial_consistency/seasonal_overview/",
#                  all_providers[i_match_prob],
#                  stringr::str_sub(i_prov_stn, prov_match[i_match_prob, 2] + 2),
#                  ext = "png")
#   
#   dir_create(path_dir(fn_out))
#   
#   file_copy(i_fn, fn_out, overwrite = T)
#   
# }
# 


# plot monthly obs vs sim ---------------------------------------------

dat_sim <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/07_SPATIAL_CONSISTENCY/SC_5REF/MonthlySim_5Ref.rds")
setDT(dat_sim)

dat_obs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/07_SPATIAL_CONSISTENCY/data-to-check/data_wide_HS.rds")
dat_obs_sim <- merge(
  dat_obs %>% melt(id.vars = c("year", "month"),
                   value.name = "obs", 
                   variable.name = "Name",
                   variable.factor = F),
  dat_sim %>% melt(id.vars = c("year", "month"), 
                   value.name = "sim", 
                   variable.name = "Name",
                   variable.factor = F),
  by = c("year", "month", "Name"),
  all = T
)

stns_to_plot <- dat_obs_sim[!is.na(sim), sort(unique(Name))]


for(i_stn in stns_to_plot){
  
  dat_plot <- dat_obs_sim[Name == i_stn & month %in% c(11, 12, 1:4)]
  dat_plot2 <- merge(dat_plot, 
                     dat_hs2[Name == i_stn & month %in% c(11, 12, 1:4)], 
                     all = T)
  
  mitmatmisc::add_month_fct(dat_plot2, 10)
  
  tit <- dat_meta[Name == i_stn, paste0(Name, ", ", Elevation, "m", ", ", 
                                        Longitude, " E", ", ", Latitude, " N")]
  
  gg <-
    dat_plot2 %>% 
    ggplot(aes(year))+
    geom_point(aes(y = HS, colour = frac_gapfilled), na.rm = T)+
    geom_line(aes(y = HS), na.rm = T, alpha = 0.5)+
    geom_point(aes(y = sim), na.rm = T, colour = "red", alpha = 0.5)+
    geom_line(aes(y = sim), na.rm = T, colour = "red", alpha = 0.5)+
    # geom_smooth(se = F, method = "lm", formula = y ~ x, colour = "grey20", na.rm = T)+
    facet_wrap(~month_fct, scales = "free_y")+
    scale_color_binned("Obs: frac gap fill",
                       type = "viridis", n.breaks = 10, limits = c(0,1))+
    theme_bw()+
    # theme(legend.position = "bottom")+
    ggtitle(tit,
            "Red transparent points/lines: statistical spatial simulation")+
    xlab(NULL)+ylab("Mean monthly HS [cm]")+
    xlim(1961, 2020)
  
  
  fn_out <- path("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/zenodo/aux_paper_spatial_consistency/monthly_time_series/",
                 dat_meta[Name == i_stn, Provider],
                 i_stn,
                 ext = "png")
  
  dir_create(path_dir(fn_out))
  
  ggsave(gg,
         filename = fn_out,
         width = 12, height = 6)
  
}




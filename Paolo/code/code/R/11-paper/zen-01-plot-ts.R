# plot monthly time series


library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(scico)

library(forcats)
library(foreach)
library(flextable)
library(officer)
library(fs)

dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/meta-with-cluster-01.rds")
dat_hs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/data_long_HS.rds")
dat_hs <- mitmatmisc::add_month_fct(dat_hs, 10)

all_provider <- sort(unique(dat_meta$Provider))

mitmatmisc::init_parallel_ubuntu(6)

foreach(
  i_prov = all_provider,
  .inorder = F
) %dopar% {
  
  all_stn <- sort(dat_meta[Provider == i_prov, Name])
  
  path_out <- path("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/zenodo/aux_paper_monthly_time_series/",
                   i_prov)
  dir_create(path_out)
  
  foreach(
    i_stn = all_stn
  ) %do% {
    
    fn_out <- path(path_out, i_stn, ext = "png")
    
    dat_plot <- dat_hs[Name == i_stn & month %in% c(11,12,1:5)]
    if(nrow(dat_plot) == 0) return(paste0("Nothing to do:", i_stn))
    
    tit <- dat_meta[Name == i_stn, paste0(Name, ", ", Elevation, "m", ", ", 
                                          Longitude, " E", ", ", Latitude, " N")]
    
    gg <- dat_plot %>% 
      ggplot(aes(year, HS))+
      geom_point(aes(colour = frac_gapfilled), na.rm = T)+
      geom_smooth(se = F, method = "lm", formula = y ~ x, colour = "grey20", na.rm = T)+
      facet_wrap(~month_fct)+
      scale_color_binned(type = "viridis", n.breaks = 10, limits = c(0,1))+
      theme_bw()+
      # theme(legend.position = "bottom")+
      ggtitle(tit)+
      xlab(NULL)+ylab("Mean monthly HS [cm]")+
      xlim(1961, 2020)
    
    
    ggsave(gg,
           filename = fn_out,
           width = 12, height = 6)
    
  }
  

  return(tit)
}


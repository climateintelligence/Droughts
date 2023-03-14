# plot all time series



library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)

# prep data ---------------------------------------------------------------



dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/r-data/meta-01-original.rds")
dat_hs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/r-data/hs-01-original.rds")
dat_hn <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/r-data/hn-01-original.rds")

dat_all <- merge(dat_hs, dat_hn, all = T)
rm(dat_hs, dat_hn)

dat_all[, month := month(date)]
dat_all[, season_year := year(date)]
dat_all[month <= 8, season_year := season_year - 1L]

# dat_all_sub <- dat_all[month >= 11 | month <= 5]



# settings ----------------------------------------------------------------

cols <- scales::hue_pal()(2)
names(cols) <- c("hs", "hn")

grp_size <- 6
month_start <- 10
month_end <- 6

all_provider <- sort(unique(dat_all$provider))
# subset provider
all_provider <- "FR_METEOFRANCE"

# plot --------------------------------------------------------------------


for(i_provider in all_provider){
  
  pdf(paste0("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/fig-timeseries/", i_provider, ".pdf"),
      width = 12, height = 6)
  
  dat_plot <- dat_all[provider == i_provider]
  
  all_stn_name <- sort(unique(dat_plot$stn_name))
  
  for(i_stn_name in all_stn_name){
    
    dat_plot2 <- dat_plot[stn_name == i_stn_name]
    dat_plot2[, grp := floor((season_year - min(season_year))/grp_size)]
    ylims <- range(c(dat_plot2$hs, dat_plot2$hn), na.rm = T)
    all_grp <- seq(min(dat_plot2$grp), max(dat_plot2$grp))
    
    # fill the data to complete panels
    all_dates <- seq(make_date(min(dat_plot2$season_year), 9, 1),
                     make_date(min(dat_plot2$season_year) + grp_size*length(all_grp), 8, 31),
                     by = "day")
    data.table(date = all_dates) %>% 
      merge(dat_plot2, all.x = T, by = "date") -> dat_plot2_full
    dat_plot2_full %>% 
      tidyr::fill(provider, stn_name, .direction = "downup") %>% 
      data.table -> dat_plot2_full
    
    dat_plot2_full[, month := month(date)]
    dat_plot2_full[, season_year := year(date)]
    dat_plot2_full[month <= 8, season_year := season_year - 1L]
    dat_plot2_full[, grp := floor((season_year - min(season_year))/grp_size)]
    
    for(i_grp in all_grp){
      
      dat_plot3 <- dat_plot2_full[grp == i_grp & (month >= month_start | month <= month_end)]
      
      gg <- ggplot(dat_plot3, aes(date))+
        geom_point(aes(y = hs, colour = "hs"), size = 1, na.rm = T)+
        geom_point(aes(y = hn, colour = "hn"), size = 0.5, na.rm = T)+
        facet_wrap(~season_year, scales = "free_x")+
        theme_bw()+
        scale_colour_manual("", values = cols)+
        scale_x_date(date_labels = "%b", date_breaks = "1 month")+
        scale_y_continuous(limits = ylims)+
        xlab(NULL)+
        ylab("HS / HN [cm]")+
        ggtitle(paste0(i_stn_name, " (", dat_meta[Name == i_stn_name, Elevation], "m)",
                       ": ", i_grp + 1, "/", length(all_grp)))
      
      print(gg)
      
    }
    
    
    
    
    
    
  }
  
  dev.off()
}






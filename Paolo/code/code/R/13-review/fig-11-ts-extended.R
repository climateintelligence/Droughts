# plots of ts with 500m averages

library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(foreach)
library(patchwork)

dat_meta_clust <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/meta-with-cluster-01.rds")
dat_month <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/data-01-monthly.rds")
dat_seasonal <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/data-02-seasonal.rds")






# month --------------------------------------------------------------------

dat_month %>% 
  merge(dat_meta_clust, by = "Name") -> dat_plot_ts
dat_plot_ts[, elev_fct := cut(Elevation, breaks = seq(0, 3000, by = 500), dig.lab = 5)]
mitmatmisc::add_month_fct(dat_plot_ts, 10)
dat_plot_ts[, HS_stn_mean := mean(HS), .(Name, month)]
dat_plot_ts[, HS_anomaly := HS - HS_stn_mean]

dat_plot_ts_mean <- dat_plot_ts[, 
                                .(meanHS = mean(HS),
                                  meanHS_anomaly = mean(HS_anomaly),
                                  nn = .N),
                                .(year, month_fct, elev_fct, cluster_fct)]

dat_plot_ts_mean[, elev_fct2 := fct_rev(elev_fct)]

cols_cluster <- setNames(scales::brewer_pal(palette = "Set1")(5),
                         levels(dat_plot_ts$cluster_fct))

pdf(file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/zenodo-SM/ts-500m-extended_monthHS.pdf",
    width = 16, height = 8)

for(i_elev in levels(dat_plot_ts$elev_fct)){
  
  dat_i_single <- dat_plot_ts[elev_fct == i_elev]
  dat_i_mean <- dat_plot_ts_mean[elev_fct == i_elev]
  
  gg <-
    dat_i_single %>% 
    ggplot(aes(year, HS))+
    geom_line(aes(group = Name), alpha = 0.2)+

    geom_line(data = dat_i_mean, 
              aes(y = meanHS, colour = cluster_fct))+
    scale_color_manual("", values = cols_cluster, guide = F)+
    scale_linetype_manual("", values = c("solid", "dashed"))+
    scale_size_manual("", values = c(1, 0.5))+
    scale_x_continuous(labels = c("'70", "'80", "'90", "2000", "'10", "'20"))+
    facet_grid(cluster_fct ~ month_fct)+
    theme_bw()+
    ggtitle("Single station (black transparent lines) and station average (coloured line) HS",
            paste0(i_elev, "m"))+
      xlab(NULL)+
      ylab("HS [cm]")
  
  
  print(gg)
  
  
}
dev.off()



# seasonalHS --------------------------------------------------------------


dat_seasonal[!startsWith(variable, "SCD")] %>% 
  merge(dat_meta_clust, by = "Name") -> dat_plot_ts
dat_plot_ts[, elev_fct := cut(Elevation, breaks = seq(0, 3000, by = 500), dig.lab = 5)]
dat_plot_ts[, value_stn_mean := mean(value), .(Name, variable)]
dat_plot_ts[, value_anomaly := value - value_stn_mean]
dat_plot_ts[, variable_fct := fct_relevel(factor(variable), "maxHS_NDJFMAM", after = Inf)]

dat_plot_ts_mean <- dat_plot_ts[, 
                                .(mean_value = mean(value),
                                  mean_value_anomaly = mean(value_anomaly),
                                  nn = .N),
                                .(year, variable, elev_fct, cluster_fct)]

dat_plot_ts_mean[, elev_fct2 := fct_rev(elev_fct)]
dat_plot_ts_mean[, variable_fct := fct_relevel(factor(variable), "maxHS_NDJFMAM", after = Inf)]



pdf(file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/zenodo-SM/ts-500m-extended_seasonalHS.pdf",
    width = 16, height = 8)

for(i_elev in levels(dat_plot_ts$elev_fct)){
  
  dat_i_single <- dat_plot_ts[elev_fct == i_elev]
  dat_i_mean <- dat_plot_ts_mean[elev_fct == i_elev]
  
  gg <-
    dat_i_single %>% 
    ggplot(aes(year, value))+
    geom_line(aes(group = Name), alpha = 0.2)+
    
    geom_line(data = dat_i_mean, 
              aes(y = mean_value, colour = cluster_fct))+
    scale_color_manual("", values = cols_cluster, guide = F)+
    scale_linetype_manual("", values = c("solid", "dashed"))+
    scale_size_manual("", values = c(1, 0.5))+
    scale_x_continuous(labels = c("'70", "'80", "'90", "2000", "'10", "'20"))+
    facet_grid(cluster_fct ~ variable_fct)+
    theme_bw()+
    ggtitle("Single station (black transparent lines) and station average (coloured line) HS",
            paste0(i_elev, "m"))+
    xlab(NULL)+
    ylab("HS [cm]")
  
  
  print(gg)
  
  
}
dev.off()


# seasonal SCD ------------------------------------------------------------



dat_seasonal[startsWith(variable, "SCD")] %>% 
  merge(dat_meta_clust, by = "Name") -> dat_plot_ts
dat_plot_ts[, elev_fct := cut(Elevation, breaks = seq(0, 3000, by = 500), dig.lab = 5)]
dat_plot_ts[, value_stn_mean := mean(value), .(Name, variable)]
dat_plot_ts[, value_anomaly := value - value_stn_mean]
dat_plot_ts[, variable_fct := fct_relevel(factor(variable), "SCD_NDJF")]

dat_plot_ts_mean <- dat_plot_ts[, 
                                .(mean_value = mean(value),
                                  mean_value_anomaly = mean(value_anomaly),
                                  nn = .N),
                                .(year, variable, elev_fct, cluster_fct)]

dat_plot_ts_mean[, elev_fct2 := fct_rev(elev_fct)]
dat_plot_ts_mean[, variable_fct := fct_relevel(factor(variable), "SCD_NDJF")]



pdf(file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/zenodo-SM/ts-500m-extended_seasonalSCD.pdf",
    width = 16, height = 8)

for(i_elev in levels(dat_plot_ts$elev_fct)){
  
  dat_i_single <- dat_plot_ts[elev_fct == i_elev]
  dat_i_mean <- dat_plot_ts_mean[elev_fct == i_elev]
  
  gg <-
    dat_i_single %>% 
    ggplot(aes(year, value))+
    geom_line(aes(group = Name), alpha = 0.2)+
    
    geom_line(data = dat_i_mean, 
              aes(y = mean_value, colour = cluster_fct))+
    scale_color_manual("", values = cols_cluster, guide = F)+
    scale_linetype_manual("", values = c("solid", "dashed"))+
    scale_size_manual("", values = c(1, 0.5))+
    scale_x_continuous(labels = c("'70", "'80", "'90", "2000", "'10", "'20"))+
    facet_grid(cluster_fct ~ variable_fct)+
    theme_bw()+
    ggtitle("Single station (black transparent lines) and station average (coloured line) SCD",
            paste0(i_elev, "m"))+
    xlab(NULL)+
    ylab("HS [cm]")
  
  
  print(gg)
  
  
}
dev.off()

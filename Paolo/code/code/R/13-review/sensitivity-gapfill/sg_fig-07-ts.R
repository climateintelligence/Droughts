# plots of ts

library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(foreach)
library(patchwork)

dat_meta_clust <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/meta-with-cluster-01.rds")
dat_month <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/sensitivity-gapfill/data-01-monthly.rds")
# dat_seasonal <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/data-02-seasonal.rds")



# month -------------------------------------------------------------------



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

# add numbers?
dat_nn <- dat_plot_ts_mean[nn > 5, 
                           .(nn = unique(nn),
                             yy = max(meanHS)),
                           .(elev_fct2, month_fct, cluster_fct)]
dat_nn[, yy_max := max(yy), elev_fct2]
xx_rename <- setNames(c(1970, 1980, 1990, 2000, 2010),
                      levels(dat_nn$cluster_fct))
dat_nn[, xx := xx_rename[cluster_fct]]

cols_manual <- setNames(scales::brewer_pal(palette = "Set1")(5),
                        levels(dat_plot_ts_mean$cluster_fct))

gg <-
dat_plot_ts_mean[nn > 5]  %>% 
  ggplot(aes(year, meanHS, colour = cluster_fct))+
  geom_line(size = 0.5, alpha = 0.8)+
  geom_text(data = dat_nn,
            aes(xx, yy_max, label = nn),
            vjust = 1, hjust = 0, size = 3, show.legend = F)+
  scale_color_manual("", values = cols_manual)+
  scale_x_continuous(labels = c("'70", "'80", "'90", "2000", "'10", "'20"))+
  facet_grid(elev_fct2 ~ month_fct, scales = "free_y")+
  theme_bw()+
  theme(panel.spacing.x = unit(9, "pt"),
        legend.position = "bottom")+
  xlab(NULL)+
  ylab("Mean HS [cm]")

ggsave(gg,
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/zenodo-SM/Figure S11.png",
       width = 12, height = 6)


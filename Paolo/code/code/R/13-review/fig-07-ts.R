# plots of ts

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

gg <-
dat_plot_ts_mean[nn > 5]  %>% 
  ggplot(aes(year, meanHS, colour = cluster_fct))+
  geom_line(size = 0.5, alpha = 0.8)+
  geom_text(data = dat_nn,
            aes(xx, yy_max, label = nn),
            vjust = 1, hjust = 0, size = 3, show.legend = F)+
  scale_color_brewer("", palette = "Set1")+
  scale_x_continuous(labels = c("'70", "'80", "'90", "2000", "'10", "'20"))+
  facet_grid(elev_fct2 ~ month_fct, scales = "free_y")+
  theme_bw()+
  theme(panel.spacing.x = unit(9, "pt"),
        legend.position = "bottom")+
  xlab(NULL)+
  ylab("Mean HS [cm]")

ggsave(gg,
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/Figure 6.png",
       width = 12, height = 6)



# dat_plot_ts_mean[nn > 5] %>% 
#   dcast(year + month_fct + elev_fct ~ cluster_fct, value.var = "meanHS") %>% 
#   .[, levels(dat_plot_ts_mean$cluster_fct), with = F] %>% #pairs
#   cor(use = "p") -> mat_cor
# 
# mat_cor[lower.tri(mat_cor)] <- NA
# data.table(reg = rownames(mat_cor), mat_cor) %>% 
#   flextable() %>% 
#   set_header_labels("reg" = "") %>% 
#   colformat_num() %>% 
#   autofit() -> ft
# 
# 
# read_docx() %>% 
#   body_add_flextable(ft) %>% 
#   print(target = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/table/supplement/ts-month-corr.docx")
# 

# by group
# 
# dat_plot_ts_mean[nn > 5] %>% 
#   dcast(year + month_fct + elev_fct ~ cluster_fct, value.var = "meanHS") %>% 
#   split(by = c("month_fct", "elev_fct"), drop = T) %>% 
#   lapply(function(x){
#     x[, levels(dat_plot_ts_mean$cluster_fct), with = F] %>% 
#       cor(use = "p") %>% 
#       symnum
#   })
#   


# seasonal HS -------------------------------------------------------------------


dat_seasonal[!startsWith(variable, "SCD")] %>% 
  merge(dat_meta_clust, by = "Name") -> dat_plot_ts
dat_plot_ts[, elev_fct := cut(Elevation, breaks = seq(0, 3000, by = 500), dig.lab = 5)]
dat_plot_ts[, value_stn_mean := mean(value), .(Name, variable)]
dat_plot_ts[, value_anomaly := value - value_stn_mean]

dat_plot_ts_mean <- dat_plot_ts[, 
                                .(mean_value = mean(value),
                                  mean_value_anomaly = mean(value_anomaly),
                                  nn = .N),
                                .(year, variable, elev_fct, cluster_fct)]

dat_plot_ts_mean[, elev_fct2 := fct_rev(elev_fct)]
dat_plot_ts_mean[, variable_fct := fct_relevel(factor(variable), "maxHS_NDJFMAM", after = Inf)]

# add numbers?
dat_nn <- dat_plot_ts_mean[nn > 5, 
                           .(nn = unique(nn),
                             yy = max(mean_value)),
                           .(elev_fct2, variable_fct, cluster_fct)]
dat_nn[, variable_mima := substr(variable_fct, 1, 3)]
dat_nn[, yy_max := max(yy), .(elev_fct2, variable_mima)]
xx_rename <- setNames(c(1970, 1980, 1990, 2000, 2010),
                      levels(dat_nn$cluster_fct))
dat_nn[, xx := xx_rename[cluster_fct]]

gg1 <-
  dat_plot_ts_mean[nn > 5 & variable_fct != "maxHS_NDJFMAM"]  %>% 
  ggplot(aes(year, mean_value, colour = cluster_fct))+
  geom_line(size = 0.5, alpha = 0.8)+
  geom_text(data = dat_nn[variable_fct != "maxHS_NDJFMAM"],
            aes(xx, yy_max, label = nn),
            vjust = 1, hjust = 0, size = 3, show.legend = F)+
  scale_color_brewer("", palette = "Set1")+
  scale_x_continuous(labels = c("'70", "'80", "'90", "2000", "'10", "'20"))+
  ylim(0, NA)+
  facet_grid(elev_fct2 ~ variable_fct, scales = "free_y")+
  theme_bw()+
  theme(panel.spacing.x = unit(9, "pt"),
        legend.position = "bottom")+
  xlab(NULL)+
  ylab("Mean HS [cm]")
  
gg2 <-
  dat_plot_ts_mean[nn > 5 & variable_fct == "maxHS_NDJFMAM"]  %>% 
  ggplot(aes(year, mean_value, colour = cluster_fct))+
  geom_line(size = 0.5, alpha = 0.8)+
  geom_text(data = dat_nn[variable_fct == "maxHS_NDJFMAM"],
            aes(xx, yy_max, label = nn),
            vjust = 1, hjust = 0, size = 3, show.legend = F)+
  scale_color_brewer("", palette = "Set1")+
  scale_x_continuous(labels = c("'70", "'80", "'90", "2000", "'10", "'20"))+
  ylim(0, NA)+
  facet_grid(elev_fct2 ~ variable_fct, scales = "free_y")+
  theme_bw()+
  theme(panel.spacing.x = unit(9, "pt"),
        legend.position = "bottom")+
  xlab(NULL)+
  ylab("Mean maxHS [cm]")


gg_out <- (gg1 | gg2) + 
  plot_layout(width = c(3,1), guides = "collect") & 
  theme(legend.position = "bottom")


ggsave(gg_out,
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/zenodo-SM/ts-seasonalHS.png",
       width = 12, height = 6)




# seasonal SCD -------------------------------------------------------------------


dat_seasonal[startsWith(variable, "SCD")] %>% 
  merge(dat_meta_clust, by = "Name") -> dat_plot_ts
dat_plot_ts[, elev_fct := cut(Elevation, breaks = seq(0, 3000, by = 500), dig.lab = 5)]
dat_plot_ts[, value_stn_mean := mean(value), .(Name, variable)]
dat_plot_ts[, value_anomaly := value - value_stn_mean]

dat_plot_ts_mean <- dat_plot_ts[, 
                                .(mean_value = mean(value),
                                  mean_value_anomaly = mean(value_anomaly),
                                  nn = .N),
                                .(year, variable, elev_fct, cluster_fct)]

dat_plot_ts_mean[, elev_fct2 := fct_rev(elev_fct)]
dat_plot_ts_mean[, variable_fct := fct_relevel(factor(variable), "SCD_NDJF")]



# add numbers?
dat_nn <- dat_plot_ts_mean[nn > 5, 
                           .(nn = unique(nn),
                             yy = max(mean_value)),
                           .(elev_fct2, variable_fct, cluster_fct)]
dat_nn[, yy_max := max(yy), .(elev_fct2)]
xx_rename <- setNames(c(1970, 1980, 1990, 2000, 2010),
                      levels(dat_nn$cluster_fct))
dat_nn[, xx := xx_rename[cluster_fct]]

gg <-
  dat_plot_ts_mean[nn > 5]  %>% 
  ggplot(aes(year, mean_value, colour = cluster_fct))+
  geom_line(size = 0.5, alpha = 0.8)+
  geom_text(data = dat_nn,
            aes(xx, yy_max, label = nn),
            vjust = 1, hjust = 0, size = 3, show.legend = F)+
  scale_color_brewer("", palette = "Set1")+
  scale_x_continuous(labels = c("'70", "'80", "'90", "2000", "'10", "'20"))+
  ylim(0, NA)+
  facet_grid(elev_fct2 ~ variable_fct, scales = "free_y")+
  theme_bw()+
  theme(panel.spacing.x = unit(9, "pt"),
        legend.position = "bottom")+
  xlab(NULL)+
  ylab("Mean SCD [days]")



ggsave(gg,
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/zenodo-SM/ts-seasonalSCD.png",
       width = 9, height = 6)



# some numbers ------------------------------------------------------------


dat_month

# plot trends (full long period)


library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(scico)

library(forcats)
library(foreach)
library(flextable)
library(officer)

pval_sym <- function(pval){
  symnum(pval, corr = FALSE,
         cutpoints = c(0,  .001,.01,.05, .1, 1),
         symbols = c("***","**","*","."," "))
}

dat_meta_clust <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/meta-with-cluster-01.rds")

load("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/sensitivity-gapfill/trends-01-1971-2019-ols-gls.rda")



# GLS plot ------------------------------------------------------


dat_plot_full <- dat_month_gls[term == "year0"] %>% 
  merge(dat_meta_clust, by = "Name")
dat_plot_full[, est_low := estimate - 1.96 * std.error]
dat_plot_full[, est_high := estimate + 1.96 * std.error]
mitmatmisc::add_month_fct(dat_plot_full, 10)

dat_plot_full


# manual limits y (elev)
dat_ylim <- dat_plot_full[, .(max_elev = max(Elevation)), .(cluster_fct)] 
setorder(dat_ylim, cluster_fct)
dat_ylim[, max_elev := c(1250, 3000, 3000, 1250)]

cols_manual <- setNames(scales::brewer_pal(palette = "Set1")(5),
                        levels(dat_plot_full$cluster_fct))

gg <- dat_plot_full %>% 
  ggplot(aes(estimate*10, Elevation, xmin = est_low*10, xmax = est_high*10, colour = cluster_fct))+
  geom_vline(xintercept = 0)+
  geom_pointrange(size = 0.1, fatten = 0.5)+
  # geom_jitter(width = 0, height = 0.3)+
  # facet_wrap(~month_fct, nrow = 2)+
  facet_grid(cluster_fct ~ month_fct, scales = "free", space = "free")+ # free_x or free
  theme_bw()+
  theme(panel.grid.minor = element_blank())+
  scale_color_manual("", values = cols_manual, guide = F)+
  scale_x_continuous(breaks = 10*seq(-4, 4, by = 2))+
  scale_y_continuous(breaks = seq(0, 3000, by = 500), limits = c(0, NA))+
  geom_blank(inherit.aes = F, data = dat_ylim, aes(x = 0, y = max_elev))+
  xlab("Linear trend in monthly mean HS [cm per decade]")+
  ylab("Elevation [m]")


ggsave(gg,
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/zenodo-SM/Figure S10.png",
       width = 10,
       height = 11)




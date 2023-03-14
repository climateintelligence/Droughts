# try interannual variability



library(nlme)
library(broom.mixed)
library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(foreach)


dat_meta_clust <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/01_submission/rds/meta-with-cluster-01.rds")
load("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/trends-01-full_1971-2019-calyear.rda")


dat_hs_full_lm[!is.na(statistic)] -> dat_lm_sub


dat_plot_full <- dat_lm_sub[term == "year0"] %>% 
  merge(dat_meta_clust, by = "Name") %>% 
  merge(dat_hs_full_lm_rsq[, .(Name, month, r.squared, sigma)], by = c("Name", "month"))
dat_plot_full[, est_low := estimate - 1.96 * std.error]
dat_plot_full[, est_high := estimate + 1.96 * std.error]
mitmatmisc::add_month_fct(dat_plot_full, 10)
dat_plot_full[, elev_fct := cut(Elevation, breaks = seq(0, 3000, by = 1000), dig.lab = 5)]

dat_plot_full



# rsq ---------------------------------------------------------------------

dat_plot_full$r.squared %>% qplot

dat_plot_full %>% 
  ggplot(aes(r.squared, colour = cluster_fct))+
  geom_freqpoly()+
  facet_wrap(~month_fct)+
  theme_bw()

dat_plot_full %>% 
  ggplot(aes(estimate, r.squared, colour = Elevation))+
  geom_point()+
  facet_grid(cluster_fct ~ month_fct)+
  theme_bw()


dat_plot_full %>% 
  ggplot(aes(r.squared))+
  geom_histogram(aes(y = ..ncount.., fill = fct_rev(elev_fct)))+
  # scale_fill_viridis_d()+
  scale_fill_brewer()+
  facet_wrap(~month_fct)+
  theme_bw()


dat_plot_full %>% 
  ggplot(aes(r.squared))+
  geom_histogram(aes(y = ..count.., fill = fct_rev(elev_fct)))+
  scale_fill_viridis_d()+
  # scale_fill_brewer()+
  facet_wrap(~month_fct, scales = "free_y")+
  theme_bw()


dat_plot_full %>% 
  ggplot(aes(r.squared))+
  geom_histogram(aes(y = ..count..), bins = 25)+
  # scale_fill_viridis_d()+
  # scale_fill_brewer()+
  # scale_color_brewer("", palette = "Set1")+
  scale_x_continuous(labels = scales::percent_format())+
  facet_grid(elev_fct~month_fct, scales = "free_y")+
  theme_bw()+
  xlab("Fraction of explained variance by trend")+
  ylab("Count (number of stations)")

ggsave(filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/review-coauthors/iv-rsq-hist.png",
       width = 12, height = 6)

# manual normalize and colors (not needed, if free scales)
summary(dat_plot_full$r.squared)
fill_cols <- scales::brewer_pal(palette = "Greys", direction = -1)(4)
rsq_breaks <- seq(0, 0.4, by = 0.01)
dat_plot_full[, rsq_fct := cut(r.squared, breaks = rsq_breaks, labels = rsq_breaks[-length(rsq_breaks)])]

dat_plot_rsq_hist <- dat_plot_full[, .(nn = .N), .(elev_fct, month_fct, rsq_fct)]
dat_plot_rsq_hist[, nn_total := sum(nn), .(month_fct)]
dat_plot_rsq_hist[, nn_frac := nn/nn_total]
dat_plot_rsq_hist[, nn_norm := nn/nn_total]

dat_plot_rsq_hist[, xx := as.numeric(as.character(rsq_fct))]

dat_plot_rsq_hist %>% 
  ggplot(aes(xx, nn_frac, fill = fct_rev(elev_fct)))+
  geom_col()+
  scale_fill_manual(values = fill_cols)+
  facet_wrap(~month_fct)+
  theme_bw()



# ts plots ----------------------------------------------------------------

dat_hs_full %>% 
  merge(dat_meta_clust, by = "Name") -> dat_plot_ts
dat_plot_ts[, elev_fct := cut(Elevation, breaks = seq(0, 3000, by = 500), dig.lab = 5)]
mitmatmisc::add_month_fct(dat_plot_ts, 10)
dat_plot_ts[, HS_stn_mean := mean(HS), .(Name, month)]
dat_plot_ts[, HS_anomaly := HS - HS_stn_mean]
# dat_plot_ts[, HS_anomaly_perc := (HS - HS_stn_mean)/HS_stn_mean] #-> zeros!
dat_plot_ts

# dat_plot_ts_mean <- dat_plot_ts[, 
#                                 .(meanHS = mean(HS),
#                                   meanHS_anomaly = mean(HS_anomaly)),
#                                 .(year, month_fct, elev_fct)]
# 
# dat_plot_ts_mean %>% 
#   ggplot(aes(year, meanHS, colour = fct_rev(elev_fct)))+
#   geom_line()+
#   scale_color_grey()+
#   facet_wrap( ~ month_fct, scales = "free_y")+
#   theme_bw()




dat_plot_ts_mean <- dat_plot_ts[, 
                                .(meanHS = mean(HS),
                                  meanHS_anomaly = mean(HS_anomaly),
                                  nn = .N),
                                .(year, month_fct, elev_fct, cluster_fct)]

dat_plot_ts_mean %>% 
  ggplot(aes(year, meanHS, colour = cluster_fct))+
  geom_line()+
  scale_color_brewer("", palette = "Set1")+
  scale_x_continuous(labels = c("'70", "'80", "'90", "2000", "'10", "'20"))+
  facet_grid(elev_fct ~ month_fct, scales = "free_y")+
  theme_bw()+
  theme(panel.spacing.x = unit(9, "pt"))+
  xlab(NULL)+
  ylab("Mean HS [cm]")+
  ggtitle("Mean HS per group (month, 500m elevation band, region)")

ggsave(filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/review-coauthors/iv-ts-mean.png",
       width = 16, height = 8)



dat_plot_ts_mean[nn > 5] %>% 
  ggplot(aes(year, meanHS, colour = cluster_fct))+
  geom_line()+
  scale_color_brewer("", palette = "Set1")+
  scale_x_continuous(labels = c("'70", "'80", "'90", "2000", "'10", "'20"))+
  facet_grid(elev_fct ~ month_fct, scales = "free_y")+
  theme_bw()+
  theme(panel.spacing.x = unit(9, "pt"))+
  xlab(NULL)+
  ylab("Mean HS [cm]")+
  ggtitle("Mean HS per group (month by 500m elevation band by region)",
          "only if > 5 stations per group")

ggsave(filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/review-coauthors/iv-ts-mean_min5.png",
       width = 16, height = 8)





# ** pdf of in-depth view -------------------------------------------------


cols_cluster <- setNames(scales::brewer_pal(palette = "Set1")(5),
                         levels(dat_plot_ts$cluster_fct))

pdf(file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/review-coauthors/iv-ts-in-depth_elev-month.pdf",
    width = 12, height = 6)
for(i_month in month.abb[c(11,12,1:5)]){
  
  for(i_elev in levels(dat_plot_ts$elev_fct)){
    
    dat_i_single <- dat_plot_ts[month_fct == i_month & elev_fct == i_elev]
    dat_i_mean <- dat_plot_ts_mean[month_fct == i_month & elev_fct == i_elev]
    
    dat_i_mean_comp <- foreach(
      i_clust = unique(dat_i_mean$cluster_fct),
      .final = rbindlist
    ) %do% {
      cbind(dat_i_mean, cluster_fct_comp = i_clust)
    }
    dat_i_mean_comp[, lty := ifelse(cluster_fct == cluster_fct_comp,
                                    "in panel", "only for comparison")]
    dat_i_single[, cluster_fct_comp := cluster_fct]
    
    gg <- dat_i_single %>% 
      ggplot(aes(year, HS))+
      geom_line(aes(group = Name), alpha = 0.2)+
      geom_line(data = dat_i_mean_comp, 
                aes(y = meanHS, colour = cluster_fct, linetype = lty, size = lty))+
      scale_color_manual("", values = cols_cluster)+
      scale_linetype_manual("", values = c("solid", "dashed"))+
      scale_size_manual("", values = c(1, 0.5))+
      scale_x_continuous(labels = c("'70", "'80", "'90", "2000", "'10", "'20"))+
      facet_wrap(~ cluster_fct_comp)+
      theme_bw()+
      ggtitle(paste0(i_month, ", ", i_elev, "m"))
    
    
    print(gg)
    
  }
  
}
dev.off()


# ** pdf by elev ----------------------------------------------------------




pdf(file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/review-coauthors/iv-ts-in-depth_elev.pdf",
    width = 16, height = 8)

for(i_elev in levels(dat_plot_ts$elev_fct)){
  
  dat_i_single <- dat_plot_ts[elev_fct == i_elev]
  dat_i_mean <- dat_plot_ts_mean[elev_fct == i_elev]
  
  dat_i_mean_comp <- foreach(
    i_clust = unique(dat_i_mean$cluster_fct),
    .final = rbindlist
  ) %do% {
    cbind(dat_i_mean, cluster_fct_comp = i_clust)
  }
  dat_i_mean_comp[, lty := ifelse(cluster_fct == cluster_fct_comp,
                                  "in panel", "only for comparison")]
  dat_i_single[, cluster_fct_comp := cluster_fct]
  
  gg <-
  dat_i_single %>% 
    ggplot(aes(year, HS))+
    geom_line(aes(group = Name), alpha = 0.2)+
    # geom_line(data = dat_i_mean_comp, 
    #           aes(y = meanHS, colour = cluster_fct, linetype = lty, size = lty))+
    geom_line(data = dat_i_mean, 
              aes(y = meanHS, colour = cluster_fct))+
    scale_color_manual("", values = cols_cluster, guide = F)+
    scale_linetype_manual("", values = c("solid", "dashed"))+
    scale_size_manual("", values = c(1, 0.5))+
    scale_x_continuous(labels = c("'70", "'80", "'90", "2000", "'10", "'20"))+
    # facet_grid(cluster_fct_comp ~ month_fct)+
    facet_grid(cluster_fct ~ month_fct)+
    theme_bw()+
    ggtitle(paste0(i_elev, "m"))
  
  
  print(gg)
  
  
}
dev.off()




# anomalies ---------------------------------------------------------------

dat_plot_ts[, .(Name, Elevation, cluster_fct, month_fct, HS_stn_mean)] %>% 
  unique  %>% 
  ggplot(aes(Elevation, HS_stn_mean, colour = cluster_fct))+
  geom_point()+
  scale_color_manual("", values = cols_cluster)+
  facet_wrap(~ month_fct, scales = "free_y")+
  theme_bw()



dat_plot_ts_mean[nn > 5] %>% 
  ggplot(aes(year, meanHS_anomaly, colour = cluster_fct))+
  geom_line()+
  scale_color_brewer("", palette = "Set1")+
  scale_x_continuous(labels = c("'70", "'80", "'90", "2000", "'10", "'20"))+
  facet_grid(elev_fct ~ month_fct, scales = "free_y")+
  theme_bw()+
  theme(panel.spacing.x = unit(9, "pt"))+
  xlab(NULL)+
  ylab("Mean HS [cm]")+
  ggtitle("Mean of HS anomalies per group (month by 500m elevation band by region)",
          "only if > 5 stations per group")





# residual error ----------------------------------------------------------

dat_plot_full %>% 
  ggplot(aes(Elevation, sigma))+
  geom_point()+
  facet_wrap(~month_fct)
# 
# 
# dat_plot_full %>% 
#   ggplot(aes(49*estimate, sigma, colour = cluster_fct))+
#   geom_abline(intercept = 0, slope = c(-1,1), linetype = "dashed")+
#   geom_point(size = 0.5, alpha = 0.5)+
#   scale_color_brewer("", palette = "Set1")+
#   facet_wrap(~month_fct)+
#   theme_bw()+
#   xlab("Trend over full period 1971-2019 [cm]")+
#   ylab("Residual standard deviation [cm]\n = interannual variability after accounting for trend")




dat_plot_full %>% 
  ggplot(aes(Elevation, 49*estimate/sigma, colour = cluster_fct))+
  # geom_abline(intercept = 0, slope = c(-1,1), linetype = "dashed")+
  geom_point(size = 0.5, alpha = 1)+
  scale_color_brewer("", palette = "Set1")+
  facet_wrap(~month_fct)+
  theme_bw()+
  xlab("Elevation[m]")+
  ylab("Trend over full period 1971-2019 divided by residual error\n ?= ratio of trend with interannual variability")

ggsave(filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/review-coauthors/iv-residual-error.png",
       width = 12, height = 8)

dat_check <- dat_plot_full[month == 5]
dat_check[, es_sig := 10*estimate/sigma]
dat_check[round(es_sig, 5) == -0.07105] %>% 
  with(plot(estimate, sigma))

# try gls -----------------------------------------------------------------

# test 1
dat_hs_full %>% 
  merge(dat_meta_clust) %>% 
  .[month == 5 & Elevation > 1500 & cluster_fct == "SW", .N, Name]

dat1 <- dat_hs_full[Name == "Fedaia" & month == 5]
dat1 <- dat_hs_full[Name == "Ritom_Piora" & month == 5]
dat1[, yearc := year - median(year)]
with(dat1, plot(yearc, HS))

# lm1 <- lm(HS ~ year0, dat1)
# plot(lm1)

gls0 <- gls(HS ~ year0, data = dat1)
plot(gls0)
gls1 <- gls(HS ~ year0, data = dat1, weights = varExp(form = ~yearc))
gls1
plot(gls1)

summary(gls0)
summary(gls1)
anova(gls0, gls1)

library(broom.mixed)
tidy(gls1)
glance(gls1)

gls1 %>% str
gls1$modelStruct$varStruct
coef(gls1$modelStruct$varStruct)

f_gls <- function(dat1){
  
  if(length(unique(dat1$HS)) <= 2) return(NULL)
  
  gls1 <- try(gls(HS ~ year0, data = dat1, weights = varExp(form = ~year0)))
  if(inherits(gls1, "try-error")) {
    gls0 <- gls(HS ~ year0, data = dat1)
    data.table(tidy(gls0),
               glance(gls0),
               cf_var_exp = NA_real_)
  } else {
    data.table(tidy(gls1),
               glance(gls1),
               cf_var_exp = coef(gls1$modelStruct$varStruct))
  }

}

dat_hs_full_gls <- dat_hs_full[,
                               f_gls(.SD), 
                               .(Name, month)]

# check missing model fits
dat_hs_full_lm[is.na(statistic)]
dat_hs_full_lm %>% 
  merge(unique(dat_hs_full_gls[, .(Name, month)])) %>% summary

dat_hs_full_lm[term == "year0"] %>% 
  merge(unique(dat_hs_full_gls[, .(Name, month, in_gls = "in")]), all.x = T) %>% 
  .[is.na(in_gls)] %>%
  .[!is.na(statistic)] %>% 
  # .[, .N, month] # -> mostly May
  .[month != 5] # -> if we adopt this, then maybe check the non-converged further...



# ** compare to lm --------------------------------------------------------

dat_comp <- merge(dat_hs_full_gls[term == "year0" & !is.na(statistic), 
                                  .(Name, month, est_gls = estimate, pval_gls = p.value,
                                    cf_var_exp, sigma)],
                  dat_hs_full_lm[term == "year0"& !is.na(statistic),  
                                 .(Name, month, est_lm = estimate, pval_lm = p.value)])
mitmatmisc::add_month_fct(dat_comp, 10)
dat_comp[, cf_var_exp_sd_rel := sqrt(exp(cf_var_exp*2*49))]

dat_comp %>% 
  ggplot(aes(est_lm, est_gls))+
  geom_point()+
  geom_abline()+
  facet_wrap(~month_fct)+
  theme_bw()

dat_comp %>% 
  ggplot(aes(pval_lm, pval_gls))+
  geom_point()+
  geom_abline()+
  facet_wrap(~month_fct)+
  theme_bw()

dat_comp %>% with(table(p_gls = pval_gls < 0.05, p_lm = pval_lm < 0.05))

dat_comp[cf_var_exp > 2]
dat_comp[cf_var_exp < 2] %>% 
  ggplot(aes(cf_var_exp))+
  geom_histogram()+
  facet_wrap(~month_fct)+
  theme_bw()

dat_comp[cf_var_exp < 2] %>% 
  merge(dat_meta_clust) %>% 
  ggplot(aes(Elevation, cf_var_exp))+
  geom_point()+
  facet_wrap(~month_fct)+
  theme_bw()

# dat_comp[cf_var_exp_sd_rel > 0.001] %>% # 12 stn difference
# dat_comp[!is.na(cf_var_exp)] %>% 
dat_comp[!is.na(cf_var_exp) & sigma < 200] %>%
  merge(dat_meta_clust) %>% 
  ggplot(aes(Elevation, cf_var_exp, colour = cluster_fct))+
  geom_hline(yintercept = 0, linetype = "dashed")+
  geom_point(alpha = 0.6)+
  # geom_smooth(se = F)+
  scale_color_brewer("", palette = "Set1", guide = F)+
  facet_wrap(~month_fct)+
  theme_bw()+
  xlab("Elevation [m]")+
  ylab("Time coefficient of residual variance")


ggsave(filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/review-coauthors/gls-var-coef-elev.png",
       width = 12, height = 8)



dat_comp[!is.na(cf_var_exp) & sigma < 200] %>%
  merge(dat_meta_clust) %>% 
  ggplot(aes(sigma, cf_var_exp, colour = cluster_fct))+
  # geom_hline(yintercept = 0, linetype = "dashed")+
  geom_point(alpha = 0.6)+
  # geom_smooth(se = F)+
  scale_color_brewer("", palette = "Set1", guide = F)+
  # facet_wrap(~month_fct)+
  facet_grid(cluster_fct ~ month_fct)+
  theme_bw()

dat_comp[sigma > 200]

dat_comp[!is.na(cf_var_exp) & sigma < 200] %>%
  merge(dat_meta_clust) %>% 
  ggplot(aes(Elevation, sigma*(cf_var_exp_sd_rel - 1), colour = cluster_fct))+
  # geom_hline(yintercept = 0, linetype = "dashed")+
  geom_point(alpha = 0.6)+
  # geom_smooth(se = F)+
  scale_color_brewer("", palette = "Set1", guide = F)+
  facet_wrap(~month_fct)+
  theme_bw()

dat_comp[!is.na(cf_var_exp) & sigma < 200] %>%
  merge(dat_meta_clust) %>% 
  ggplot(aes(est_gls*10, cf_var_exp, colour = cluster_fct))+
  geom_hline(yintercept = 0, linetype = "dashed")+
  geom_vline(xintercept = 0, linetype = "dashed")+
  geom_point(alpha = 0.6)+
  scale_color_brewer("", palette = "Set1", guide = F)+
  facet_wrap(~month_fct, scales = "free")+
  theme_bw()+
  xlab("Trend HS [cm/decade]")+
  ylab("Time coefficient of residual variance")

ggsave(filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/review-coauthors/gls-var-coef-trend.png",
       width = 12, height = 8)


# not by elevation but meanHS as proxy
dat_comp[!is.na(cf_var_exp) & sigma < 200] %>% 
  merge(dat_hs_full_lm[term == "(Intercept)"& !is.na(statistic),  
                       .(Name, month, est_int = estimate)]) %>% 
  merge(dat_meta_clust) %>% 
  .[est_int > 1] %>% # subset to meaningful
  ggplot(aes(est_int, cf_var_exp, colour = cluster_fct))+
  geom_hline(yintercept = 0, linetype = "dashed")+
  geom_point(alpha = 0.6)+
  # geom_smooth(se = F)+
  scale_color_brewer("", palette = "Set1", guide = F)+
  facet_wrap(~month_fct, scales = "free_x")+
  theme_bw()+
  xlab("meanHS at start of period [cm], only >1cm shown")+
  ylab("Time coefficient of residual variance")

ggsave(filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/review-coauthors/gls-var-coef-meanHS.png",
       width = 12, height = 8)


# relative
# dat_comp[!is.na(cf_var_exp) & sigma < 200] %>%
#   merge(dat_meta_clust) %>% 
#   ggplot(aes(est_gls*10, cf_var_exp_sd_rel, colour = cluster_fct))+
#   geom_hline(yintercept = 1, linetype = "dashed")+
#   geom_vline(xintercept = 0, linetype = "dashed")+
#   geom_point(alpha = 0.6)+
#   # geom_smooth(se = F)+
#   scale_color_brewer("", palette = "Set1", guide = F)+
#   scale_y_log10()+
#   facet_wrap(~month_fct, scales = "free")+
#   theme_bw()+
#   xlab("Trend HS [cm/decade]")+
#   ylab("Time coefficient of residual variance")



# # as fraction change start to end
# dat_comp[cf_var_exp < 2 & cf_var_exp_sd_rel > 0.001] %>% 
#   merge(dat_meta_clust) %>% 
#   ggplot(aes(Elevation, sqrt(exp(cf_var_exp*2*49)), colour = cluster_fct))+
#   geom_hline(yintercept = 1, linetype = "dashed")+
#   geom_point(alpha = 0.6)+
#   # geom_smooth(se = F)+
#   scale_color_brewer("", palette = "Set1", guide = F)+
#   scale_y_log10(labels = scales::trans_format("log10", scales::math_format()))+
#   facet_wrap(~month_fct)+
#   theme_bw()+
#   xlab("Elevation [m]")+
#   ylab("Fraction of residual standard deviation 2019 compared to 1971 \n (note the log-scale!)")
# 
# ggsave(filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/review-coauthors/gls-var-coef-frac.png",
#        width = 12, height = 8)

# ** constPower varfunc ----------------------------------------------------


# f_gls2 <- function(dat1){
#   
#   gls1 <- try(gls(HS ~ year0, data = dat1, weights = varConstPower(form = ~year0),
#                   control = glsControl(maxIter = 200, msMaxIter = 200)))
#   if(inherits(gls1, "try-error")) return(NULL)
#   data.table(tidy(gls1),
#              glance(gls1),
#              coef(gls1$modelStruct$varStruct))
# }
# 
# dat_hs_full_gls2 <- dat_hs_full[,
#                                f_gls2(.SD), 
#                                .(Name, month)]

# -> convergence issues



# gls but check reduce to normal ------------------------------------------



# test 1
dat_hs_full %>% 
  merge(dat_meta_clust) %>% 
  .[month == 5 & Elevation > 1500 & cluster_fct == "SW", .N, Name]

dat1 <- dat_hs_full[Name == "Fedaia" & month == 5]
dat1 <- dat_hs_full[Name == "Ritom_Piora" & month == 5]
dat1[, yearc := year - median(year)]
with(dat1, plot(yearc, HS))

# lm1 <- lm(HS ~ year0, dat1)
# plot(lm1)

gls0 <- gls(HS ~ year0, data = dat1)
plot(gls0)
gls1 <- gls(HS ~ year0, data = dat1, weights = varExp(form = ~year0))
gls1
plot(gls1)

summary(gls0)
summary(gls1)
anova(gls0, gls1)


f_gls3 <- function(dat1){
  
  if(length(unique(dat1$HS)) <= 2) return(NULL)
  
  gls0 <- gls(HS ~ year0, data = dat1)
  gls1 <- try(gls(HS ~ year0, data = dat1, weights = varExp(form = ~year0)))
  if(inherits(gls1, "try-error")) {
    data.table(tidy(gls0),
               glance(gls0),
               cf_var_exp = NA_real_,
               pval_comp = NA_real_)
  } else {
    an <- anova(gls0, gls1)
    pval_comp <- an[2, "p-value"]
    if(pval_comp < 0.1){
      data.table(tidy(gls1),
                 glance(gls1),
                 cf_var_exp = coef(gls1$modelStruct$varStruct),
                 pval_comp = pval_comp)
    } else {
      data.table(tidy(gls0),
                 glance(gls0),
                 cf_var_exp = NA_real_,
                 pval_comp = pval_comp)
    }
  }

}

dat_hs_full_gls3 <- dat_hs_full[,
                                f_gls3(.SD), 
                                .(Name, month)]

# check missing model fits
dat_hs_full_lm[is.na(statistic)]


dat_hs_full_lm[term == "year0"] %>% 
  merge(unique(dat_hs_full_gls3[, .(Name, month, in_gls = "in")]), all.x = T) %>% 
  .[is.na(in_gls)] %>%
  .[!is.na(statistic)]


# ** compare to lm --------------------------------------------------------

dat_comp <- merge(dat_hs_full_gls3[term == "year0" & !is.na(statistic), 
                                  .(Name, month, est_gls = estimate, pval_gls = p.value,
                                    cf_var_exp)],
                  dat_hs_full_lm[term == "year0"& !is.na(statistic),  
                                 .(Name, month, est_lm = estimate, pval_lm = p.value)])
mitmatmisc::add_month_fct(dat_comp, 10)
dat_comp[, cf_var_exp_sd_rel := sqrt(exp(cf_var_exp*2*49))]


dat_comp %>% 
  ggplot(aes(est_lm, est_gls))+
  geom_point()+
  geom_abline()+
  facet_wrap(~month_fct)+
  theme_bw()

# -> maybe check the ones with large differences!

dat_comp %>% 
  ggplot(aes(pval_lm, pval_gls))+
  geom_point()+
  geom_abline()+
  facet_wrap(~month_fct)+
  theme_bw()

dat_comp %>% with(table(p_gls = pval_gls < 0.05, p_lm = pval_lm < 0.05))

dat_comp %>% 
  ggplot(aes(cf_var_exp))+
  geom_histogram()+
  facet_wrap(~month_fct)+
  theme_bw()


dat_comp[cf_var_exp_sd_rel > 0.001] %>% 
  merge(dat_meta_clust) %>% 
  ggplot(aes(Elevation, cf_var_exp, colour = cluster_fct))+
  geom_hline(yintercept = 0, linetype = "dashed")+
  geom_point(alpha = 0.6)+
  # geom_smooth(se = F)+
  scale_color_brewer("", palette = "Set1", guide = F)+
  facet_wrap(~month_fct)+
  theme_bw()+
  xlab("Elevation [m]")+
  ylab("Time coefficient of residual variance")


ggsave(filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/review-coauthors/gls-var-coef_modsel.png",
       width = 12, height = 8)


# as fraction change start to end
dat_comp[!is.na(cf_var_exp)] %>% 
  merge(dat_meta_clust) %>% 
  ggplot(aes(Elevation, cf_var_exp_sd_rel, colour = cluster_fct))+
  geom_hline(yintercept = 1, linetype = "dashed")+
  geom_point(alpha = 0.6)+
  # geom_smooth(se = F)+
  scale_color_brewer("", palette = "Set1", guide = F)+
  scale_y_log10(limits = c(0.001, NA),
                labels = scales::trans_format("log10", scales::math_format()))+
  facet_wrap(~month_fct)+
  theme_bw()+
  xlab("Elevation [m]")+
  ylab("Fraction of residual standard deviation 2019 compared to 1971 \n (note the log-scale!)")

ggsave(filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/review-coauthors/gls-var-coef-frac_modsel.png",
       width = 12, height = 8)

# check extreme
dat_comp[cf_var_exp_sd_rel < 0.01 | cf_var_exp_sd_rel > 10] %>% 
  merge(dat_meta_clust) -> dat_check
dat_check[, est_gls := round(est_gls, 4)]
dat_check[, est_lm := round(est_lm, 4)]


# check relation to trend




# eof ---------------------------------------------------------------------



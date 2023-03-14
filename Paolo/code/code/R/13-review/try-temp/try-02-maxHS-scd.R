# try maxHS SCD stuff


library(nlme)
library(broom.mixed)
library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(foreach)


dat_meta_clust <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/01_submission/rds/meta-with-cluster-01.rds")

# calc indices ------------------------------------------------------------

month_sub <- c(11:12, 1:5)
min_frac_avail <- 0.9


dat_hs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/gapfill-01-all.rds")
dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1961-2020/meta_long_HN_HS.rds")

# sub to paper stns
dat_lm_paper <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/01_submission/rds/trends-03-full_1971-2019-calyear.rds")
stns_sub <- unique(dat_lm_paper$Name)
dat_hs_sub <- dat_hs[Name %in% stns_sub]


setnames(dat_hs_sub, "Date", "date")
dat_hs_sub <- mitmatmisc::add_hydro_year(dat_hs_sub, 10)

dat_maxHS <- dat_hs_sub[month(date) %in% month_sub, 
                        .(maxHS = max(HS, na.rm = T),
                          nn_maxHS = sum(!is.na(HS))),
                        .(Name, hydro_year)]
max_days <- 30+31+31+28+31+30+31
dat_maxHS[nn_maxHS < min_frac_avail * max_days, maxHS := NA]

dat_SCD_fall <- dat_hs_sub[month(date) %in% c(11,12,1), 
                           .(SCD_fall = sum(HS >= 1, na.rm = T),
                             nn_SCD_fall = sum(!is.na(HS))),
                           .(Name, hydro_year)]
max_days <- 30+31+31
dat_SCD_fall[nn_SCD_fall < min_frac_avail * max_days, SCD_fall := NA]


dat_SCD_spring <- dat_hs_sub[month(date) %in% c(2:5), 
                             .(SCD_spring = sum(HS >= 1, na.rm = T),
                               nn_SCD_spring = sum(!is.na(HS))),
                             .(Name, hydro_year)]
max_days <- 28+31+30+31
dat_SCD_spring[nn_SCD_spring < min_frac_avail * max_days, SCD_spring := NA]




# subset to full period ---------------------------------------------------

# old
dat_hs_month <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/data_long_HS.rds")
dat_hs_month2 <- dat_hs_month[year >= 1971 & year <= 2019 & month %in% month_sub & !is.na(HS)]
with(dat_hs_month2, table(year, month))


# new
dat_check <- dat_maxHS
dat_check <- dat_SCD_fall
dat_check <- dat_SCD_spring

dat_check2 <- na.omit(dat_check)
dat_check2 <- dat_check2[hydro_year >= 1971 & hydro_year <= 2018]
with(dat_check2, table(hydro_year))

dat_check2[, nn_year := .N, .(Name)]
dat_check2_full <- dat_check2[nn_year == max(nn_year)]

dat_meta[Name %in% dat_check2_full$Name, .N, Provider]

# -> hmm 2019 has almost 100 stns less than 2018...
# but all from AT, so no worries.

f_full <- function(dat_check){
  dat_check2 <- na.omit(dat_check)
  dat_check2 <- dat_check2[hydro_year >= 1971 & hydro_year <= 2019]
  dat_check2[, nn_year := .N, .(Name)]
  dat_check2[nn_year == max(nn_year)]
}

dat_maxHS_full <- f_full(dat_maxHS)
dat_SCD_fall_full <- f_full(dat_SCD_fall)
dat_SCD_spring_full <- f_full(dat_SCD_spring)



# variability plots -------------------------------------------------------


dat_plot_maxHS <- dat_maxHS_full %>% merge(dat_meta_clust, by = "Name")
dat_plot_maxHS[, elev_fct := cut(Elevation, breaks = seq(0, 3000, by = 500), dig.lab = 5)]

dat_plot_maxHS_mean <- dat_plot_maxHS[, 
                                      .(mean_maxHS = mean(maxHS),
                                        nn = .N),
                                      .(hydro_year, elev_fct, cluster_fct)]


gg_maxhs <- dat_plot_maxHS_mean[nn > 5] %>% 
  ggplot(aes(hydro_year, mean_maxHS, colour = cluster_fct))+
  geom_line()+
  scale_color_brewer("", palette = "Set1")+
  scale_x_continuous(labels = c("'70", "'80", "'90", "2000", "'10", "'20"))+
  facet_grid(. ~ elev_fct)+
  theme_bw()+
  # theme(panel.spacing.x = unit(9, "pt"))+
  xlab(NULL)+
  ylab("Maximum Nov-May HS [cm]")


# 
# dat_plot_maxHS %>% 
#   ggplot(aes(hydro_year, maxHS))+
#   geom_line(aes(group = Name), alpha = 0.2)+
#   geom_line(data = dat_plot_maxHS_mean, aes(y = mean_maxHS, colour = cluster_fct))+
#   scale_color_brewer("", palette = "Set1")+
#   scale_x_continuous(labels = c("'70", "'80", "'90", "2000", "'10", "'20"))+
#   # facet_grid(elev_fct ~ cluster_fct, scales = "free_y")+
#   facet_wrap(~ elev_fct + cluster_fct, scales = "free_y")+
#   theme_bw()
# 



dat_plot_SCD_fall <- dat_SCD_fall_full %>% merge(dat_meta_clust, by = "Name")
dat_plot_SCD_fall[, elev_fct := cut(Elevation, breaks = seq(0, 3000, by = 500), dig.lab = 5)]

dat_plot_SCD_fall_mean <- dat_plot_SCD_fall[, 
                                            .(mean_SCD_fall = mean(SCD_fall),
                                              nn = .N),
                                            .(hydro_year, elev_fct, cluster_fct)]


gg_scd_fall <- dat_plot_SCD_fall_mean[nn > 5] %>% 
  ggplot(aes(hydro_year, mean_SCD_fall, colour = cluster_fct))+
  geom_line()+
  scale_color_brewer("", palette = "Set1")+
  scale_x_continuous(labels = c("'70", "'80", "'90", "2000", "'10", "'20"))+
  facet_grid(. ~ elev_fct)+
  theme_bw()+
  # theme(panel.spacing.x = unit(9, "pt"))+
  xlab(NULL)+
  ylab("SCD Nov-Jan [days]")





dat_plot_SCD_spring <- dat_SCD_spring_full %>% merge(dat_meta_clust, by = "Name")
dat_plot_SCD_spring[, elev_fct := cut(Elevation, breaks = seq(0, 3000, by = 500), dig.lab = 5)]

dat_plot_SCD_spring_mean <- dat_plot_SCD_spring[, 
                                                .(mean_SCD_spring = mean(SCD_spring),
                                                  nn = .N),
                                                .(hydro_year, elev_fct, cluster_fct)]


gg_scd_spring <- dat_plot_SCD_spring_mean[nn > 5] %>% 
  ggplot(aes(hydro_year, mean_SCD_spring, colour = cluster_fct))+
  geom_line()+
  scale_color_brewer("", palette = "Set1")+
  scale_x_continuous(labels = c("'70", "'80", "'90", "2000", "'10", "'20"))+
  facet_grid(. ~ elev_fct)+
  theme_bw()+
  # theme(panel.spacing.x = unit(9, "pt"))+
  xlab(NULL)+
  ylab("SCD Feb-May [days]")


gg_out <- gg_scd_fall / gg_maxhs / gg_scd_spring + 
  plot_layout(guides = "collect")+
  plot_annotation(title = "Mean SCD / maxHS per group (500m elevation band by region)",
                  subtitle = "only if > 5 stations per group")


ggsave(gg_out,
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/review-coauthors/maxHS-SCD-ts.png",
       width = 16, height = 8)



# trends and trend plots --------------------------------------------------

dat_full <- rbind(
  dat_maxHS_full[, .(Name, hydro_year, vv = "maxHS", value = maxHS)],
  dat_SCD_fall_full[, .(Name, hydro_year, vv = "SCD_fall", value = SCD_fall)],
  dat_SCD_spring_full[, .(Name, hydro_year, vv = "SCD_spring", value = SCD_spring)]
)

dat_full[, year0 := hydro_year - min(hydro_year)]
dat_lm_aux <- dat_full[,
                       broom::tidy(lm(value ~ year0)), 
                       .(Name, vv)]


dat_plot_lm_aux <- dat_lm_aux[term == "year0"] %>% 
  merge(dat_meta_clust, by = "Name")
dat_plot_lm_aux[, est_low := estimate - 1.96 * std.error]
dat_plot_lm_aux[, est_high := estimate + 1.96 * std.error]


dat_plot_lm_aux

# manual limits y (elev)
dat_ylim <- dat_plot_lm_aux[, .(max_elev = max(Elevation)), .(cluster_fct)] 
setorder(dat_ylim, cluster_fct)
dat_ylim[, max_elev := c(1250, 1250, 3000, 3000, 1250)]

dat_plot_lm_aux %>% 
  ggplot(aes(10*estimate, Elevation, xmin = 10*est_low, xmax = 10*est_high, colour = cluster_fct))+
  geom_vline(xintercept = 0)+
  geom_pointrange(size = 0.1, fatten = 0.5)+
  # geom_jitter(width = 0, height = 0.3)+
  # facet_wrap(~month_fct, nrow = 2)+
  facet_grid(cluster_fct ~ vv, scales = "free", space = "free_y")+ # free_x or free
  theme_bw()+
  theme(panel.grid.minor = element_blank())+
  scale_color_brewer("", palette = "Set1", guide = F)+
  # scale_x_continuous(breaks = 10*seq(-4, 4, by = 2))+
  scale_y_continuous(breaks = seq(0, 3000, by = 500), limits = c(0, NA))+
  geom_blank(inherit.aes = F, data = dat_ylim, aes(x = 0, y = max_elev))+
  xlab("Linear trend in maxHS, SCD_fall, SCD_spring [cm/days per decade]")+
  ylab("Elevation [m]")


ggsave(filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/review-coauthors/maxHS-SCD-trends.png",
       width = 10,
       height = 10)



# gls? probably not needed ------------------------------------------------


f_gls3 <- function(dat1){
  
  if(length(unique(dat1$value)) <= 2) return(NULL)
  
  gls0 <- gls(value ~ year0, data = dat1)
  gls1 <- try(gls(value ~ year0, data = dat1, weights = varExp(form = ~year0)))
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
# 
# dat_gls3 <- dat_full[,
#                      f_gls3(.SD), 
#                      .(Name, vv)]

# -> NO!



# scatter monthly mean vs max/SCD -----------------------------------------

dat_lm_paper
dat_lm_paper <- mitmatmisc::add_month_fct(dat_lm_paper, 10)
dat_meta_clust[, elev_fct := cut(Elevation, breaks = seq(0, 3000, by = 500), dig.lab = 5)]

dat_lm_aux[term == "year0" & vv == "maxHS", .(Name, est_aux = estimate)] %>% 
  merge(dat_lm_paper[term == "year0", .(Name, month_fct, est_month = estimate)]) %>% 
  merge(dat_meta_clust) -> dat_plot_comp_maxHS

dat_plot_comp_maxHS %>% 
  ggplot(aes(10*est_month, 10*est_aux, colour = cluster_fct))+
  geom_abline(linetype = "dashed")+
  geom_point(alpha = 0.5)+
  scale_color_brewer("", palette = "Set1")+
  facet_grid(elev_fct ~ month_fct, scales = "free")+
  theme_bw()+
  # coord_equal()+
  xlab("Trend mean monthly HS [cm/decade]")+
  ylab("Trend maxHS Nov-May [cm/decade]")+
  ggtitle("Comparison between trends in mean monthly HS and maximum seasonal HS",
          "dashed line is 1:1, x and y axes differ by row and column")


ggsave(filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/review-coauthors/maxHS-SCD-comp-monthly-maxHS.png",
       width = 16,
       height = 8)




dat_lm_aux[term == "year0" & vv == "SCD_fall", .(Name, est_aux = estimate)] %>% 
  merge(dat_lm_paper[term == "year0", .(Name, month_fct, est_month = estimate)]) %>% 
  merge(dat_meta_clust) -> dat_plot_comp_SCD_fall

dat_plot_comp_SCD_fall %>% 
  ggplot(aes(10*est_month, 10*est_aux, colour = cluster_fct))+
  geom_abline(linetype = "dashed")+
  geom_hline(yintercept = 0, linetype = "dashed")+
  geom_vline(xintercept = 0, linetype = "dashed")+
  geom_point(alpha = 0.5)+
  scale_color_brewer("", palette = "Set1")+
  # facet_grid(elev_fct ~ month_fct, scales = "free")+
  facet_wrap(~ elev_fct + month_fct, scales = "free", nrow = 6)+
  theme_bw()+
  # coord_equal()+
  xlab("Trend mean monthly HS [cm/decade]")+
  ylab("Trend SCD Nov-Jan [days/decade]")+
  ggtitle("Comparison between trends in mean monthly HS and fall SCD (Nov-Jan)",
          "dashed lines are x=y, x=0 and y=0; x and y axes differ in each panel")


ggsave(filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/review-coauthors/maxHS-SCD-comp-monthly-SCD_fall.png",
       width = 16,
       height = 12)




dat_lm_aux[term == "year0" & vv == "SCD_spring", .(Name, est_aux = estimate)] %>% 
  merge(dat_lm_paper[term == "year0", .(Name, month_fct, est_month = estimate)]) %>% 
  merge(dat_meta_clust) -> dat_plot_comp_SCD_spring

dat_plot_comp_SCD_spring %>% 
  ggplot(aes(10*est_month, 10*est_aux, colour = cluster_fct))+
  geom_abline(linetype = "dashed")+
  geom_hline(yintercept = 0, linetype = "dashed")+
  geom_vline(xintercept = 0, linetype = "dashed")+
  geom_point(alpha = 0.5)+
  scale_color_brewer("", palette = "Set1")+
  # facet_grid(elev_fct ~ month_fct, scales = "free")+
  facet_wrap(~ elev_fct + month_fct, scales = "free", nrow = 6)+
  theme_bw()+
  # coord_equal()+
  xlab("Trend mean monthly HS [cm/decade]")+
  ylab("Trend SCD Feb-May [days/decade]")+
  ggtitle("Comparison between trends in mean monthly HS and spring SCD (Feb-May)",
          "dashed lines are x=y, x=0 and y=0; x and y axes differ in each panel")


ggsave(filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/review-coauthors/maxHS-SCD-comp-monthly-SCD_spring.png",
       width = 16,
       height = 12)



# ts plots ----------------------------------------------------------------

pdf("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/review-coauthors/ts-maxHS.pdf",
    width = 8, height = 6)
for(i_stn in sort(unique(dat_maxHS_full$Name))){
  
  tit <- dat_meta[Name == i_stn, paste0(Name, ", ", Elevation, "m")]
  
  gg <- dat_maxHS_full[Name == i_stn] %>% 
    ggplot(aes(hydro_year, maxHS))+
    geom_point()+
    geom_smooth(method = lm, se = F, formula = y ~ x)+
    theme_bw()+
    xlab(NULL)+
    ylab("maxHS [cm]")+
    ggtitle(tit)
  
  print(gg)
}
dev.off()


pdf("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/review-coauthors/ts-SCD-fall.pdf",
    width = 8, height = 6)
for(i_stn in sort(unique(dat_SCD_fall_full$Name))){
  
  tit <- dat_meta[Name == i_stn, paste0(Name, ", ", Elevation, "m")]
  
  gg <- dat_SCD_fall_full[Name == i_stn] %>% 
    ggplot(aes(hydro_year, SCD_fall))+
    geom_point()+
    geom_smooth(method = lm, se = F, formula = y ~ x)+
    theme_bw()+
    xlab(NULL)+
    ylab("SCD Nov-Jan [days]")+
    ggtitle(tit)
  
  print(gg)
}
dev.off()


pdf("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/review-coauthors/ts-SCD-spring.pdf",
    width = 8, height = 6)
for(i_stn in sort(unique(dat_SCD_spring_full$Name))){
  
  tit <- dat_meta[Name == i_stn, paste0(Name, ", ", Elevation, "m")]
  
  gg <- dat_SCD_spring_full[Name == i_stn] %>% 
    ggplot(aes(hydro_year, SCD_spring))+
    geom_point()+
    geom_smooth(method = lm, se = F, formula = y ~ x)+
    theme_bw()+
    xlab(NULL)+
    ylab("SCD Feb-May [days]")+
    ggtitle(tit)
  
  print(gg)
}
dev.off()




# try log neg.bin for SCD -------------------------------------------------

library(MASS)
library(betareg)
# SCD_fall, Diga_Gabiet, p 112

d1 <- dat_SCD_fall_full[Name == "Diga_Gabiet"]
d1 <- dat_SCD_fall_full[Name == "Diex"]

d1[, SCD_fall_01 := SCD_fall / 92]
# transformation after betareg vignette, if 0,1 included
d1[, SCD_fall_01t := (SCD_fall_01 * (.N - 1) + 0.5)/.N]
d1
with(d1, plot(hydro_year, SCD_fall))

lm1 <- lm(SCD_fall ~ hydro_year, data = d1)
summary(lm1)
plot(lm1)

nb1 <- glm.nb(SCD_fall ~ hydro_year, data = d1)
nb1 %>% summary
plot(nb1)

br1 <- betareg(SCD_fall_01t ~ hydro_year, data = d1)
summary(br1)
plot(br1)

br2 <- betareg(SCD_fall_01t ~ hydro_year | hydro_year, data = d1)
summary(br2)
plot(br2, 1)

lmtest::lrtest(br1, br2)

# poisson or quasipoisson
# bah



# check problematic series SCD --------------------------------------------

dat_nn <- dat_SCD_fall_full[, 
                  .(n_max = sum(SCD_fall == 92),
                    n_min = sum(SCD_fall == 0)),
                  .(Name)]
dat_nn$n_max %>% qplot
dat_nn$n_min %>% qplot

dat_nn %>% 
  merge(dat_meta_clust, by = "Name") %>% 
  ggplot(aes(Elevation, n_max))+
  geom_point()

dat_nn %>% 
  merge(dat_meta_clust, by = "Name") %>% 
  ggplot(aes(Elevation, n_min))+
  geom_point()



dat_nn2 <- dat_SCD_spring_full[, 
                            .(n_max = sum(SCD_spring >= 120),
                              n_min = sum(SCD_spring == 0)),
                            .(Name)]
dat_nn2$n_max %>% qplot
dat_nn2$n_min %>% qplot

dat_nn2 %>% 
  merge(dat_meta_clust, by = "Name") %>% 
  ggplot(aes(Elevation, n_max))+
  geom_point()

dat_nn2 %>% 
  merge(dat_meta_clust, by = "Name") %>% 
  ggplot(aes(Elevation, n_min))+
  geom_point()

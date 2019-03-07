/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Version Sep/22/2014
   Zhenxing Wang (KU)
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(conp,FixConp)

#else

#ifndef LMP_FIX_CONP_H
#define LMP_FIX_CONP_H

#include "fix.h"

namespace LAMMPS_NS {

class FixConp : public Fix {
 public:
  FixConp(class LAMMPS *, int, char **);
  ~FixConp();
  int setmask();
  void init();
  void setup(int);
  void pre_force(int);
  void force_cal(int);
  void a_cal();
  void a_read();
  void b_cal();
  void equation_solve();
  void update_charge();
  int electrode_check(int);
  void sincos_a(double **);
  void sincos_b();
  void cg();
  void inv();
  void coul_cal(int, double *,int *);

 private:
  int me,runstage;
  double Btime,Btime1,Btime2;
  double Ctime,Ctime1,Ctime2;
  double Ktime,Ktime1,Ktime2;
  double cgtime,cgtime1,cgtime2;
  FILE *outf,*outa,*a_matrix_fp;
  int a_matrix_f;
  int minimizer;
  double vL,vR;
  int molidL,molidR;
  int maxiter;
  double tolerance;

  double rms(int,double,bigint,double);
  void coeffs();

  double unitk[3];
  double *ug;
  double g_ewald,eta,gsqmx,volume,slab_volfactor;
  int *kxvecs,*kyvecs,*kzvecs;
  double ***cs,***sn,**csk,**snk;
  int kmax,kmax3d,kmax_created,kcount;
  int kxmax,kymax,kzmax;
  double *sfacrl,*sfacrl_all,*sfacim,*sfacim_all;
  int everynum;
  int elenum,elenum_old,elenum_all;
  double *eleallq;
  double *aaa_all,*bbb_all;
  int *tag2eleall,*eleall2tag,*curr_tag2eleall,*ele2tag;
};

}

#endif
#endif

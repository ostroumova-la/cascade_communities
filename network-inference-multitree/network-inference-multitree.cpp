#include "stdafx.h"

int main(int argc, char* argv[]) {
  Env = TEnv(argc, argv, TNotify::StdNotify);
  Env.PrepArgs(TStr::Fmt("\nNetwork Inference Multitree. build: %s, %s. Time: %s", __TIME__, __DATE__, TExeTm::GetCurTm()));
  TExeTm ExeTm;
  Try
  const TStr InFNm  = Env.GetIfArgPrefixStr("-i:", "example-cascades.txt", "Input cascades (one file)");
  const TStr GroundTruthFNm = Env.GetIfArgPrefixStr("-n:", "example-network.txt", "Input ground-truth network (one file)");
  const TStr OutFNm  = Env.GetIfArgPrefixStr("-o:", "network", "Output file name(s) prefix");
  const TStr Iters  = Env.GetIfArgPrefixStr("-e:", "5", "Number of iterations");
  const double alpha = Env.GetIfArgPrefixFlt("-a:", 1.0, "Alpha for transmission model");
  const double Delta = Env.GetIfArgPrefixFlt("-d:", 1.0, "Delta for power law");
  const int NCascades = Env.GetIfArgPrefixInt("-nc:", -1, "Number of cascades to use (-1 indicastes all");
  const int Model = Env.GetIfArgPrefixInt("-m:", 0, "0:exponential, 1:power law, 2:rayleigh");
  const int TakeAdditional = Env.GetIfArgPrefixInt("-s:", 1, "How much additional files to create?\n\
  1:info about each edge, 2:objective function value (+upper bound), 3:Precision-recall plot, 4:ROC plot, 5:all-additional-files (default:1)\n");

  bool ComputeBound = false, ComputeInfo = false; bool CompareGroundTruth = false; bool ComputeROC = false;
  switch (TakeAdditional) {
  case 1 : ComputeInfo = true; break;
  case 2 : ComputeBound = true; break;
  case 3 : CompareGroundTruth = true; break;
  case 4 : ComputeROC = true; break;
  case 5 :
	  ComputeInfo = true;
	  // ComputeBound = true;
	  CompareGroundTruth = true; break;
  default: FailR("Bad -s: parameter.");
  }

  TNIMT NIMT(ComputeBound, CompareGroundTruth||ComputeROC);
  printf("\nLoading input cascades: %s\n", InFNm.CStr());

  // load cascade from file
  TFIn FIn(InFNm);
  NIMT.LoadCascadesTxt(FIn, Model, alpha, Delta, NCascades);

  // load ground truth network
  if (CompareGroundTruth || ComputeROC) {
	  TFIn FInG(GroundTruthFNm);
	  NIMT.LoadGroundTruthTxt(FInG);
  }

  NIMT.Init();
  printf("cascades:%d nodes:%d potential edges:%d\nRunning NETINF...\n", NIMT.GetCascs(), NIMT.GetNodes(), NIMT.CascPerEdge.Len());
  NIMT.GreedyOpt(Iters.GetInt());

  // plots showing precision/recall & accuracy using groundtruth
  if (CompareGroundTruth) {
	  TGnuPlot::PlotValV(NIMT.PrecisionRecall, TStr::Fmt("%s-precision-recall", OutFNm.CStr()), "Precision Recall", "Recall",
						 "Precision", gpsAuto, false, gpwLinesPoints, false);

	  TGnuPlot::PlotValV(NIMT.Accuracy, TStr::Fmt("%s-accuracy", OutFNm.CStr()), "Accuracy", "Iteration",
	  	  						 "Accuracy", gpsAuto, false, gpwLinesPoints);
  }
  
  // compute & store ROC 
  if (ComputeROC || CompareGroundTruth) {
    double ROC = 0;
    
    for (int i=1; i<NIMT.PrecisionRecall.Len(); i++) {
      ROC += (NIMT.PrecisionRecall[i].Val1.Val-NIMT.PrecisionRecall[i-1].Val1.Val) * NIMT.PrecisionRecall[i].Val2.Val;
    }
    
    printf("ROC: %f\n", ROC);
    
    TFOut FOutROC(TStr::Fmt("%s-roc", OutFNm.CStr()));
    FOutROC.PutStr(TStr::Fmt("%f\n", ROC));
  }

  // plot objective function
  if (ComputeBound) {
	  TFltPrV Gains;
	  for (int i=0; i<NIMT.EdgeInfoH.Len(); i++)
		  Gains.Add(TFltPr((double)(i+1), NIMT.EdgeInfoH[i].MarginalGain));

	  TGnuPlot::PlotValV(Gains, TStr::Fmt("%s-objective", OutFNm.CStr()), "Objective Function", "Iters", "Objective Function");
  }

  // save network in plain text
  // NIMT.SavePlaneTextNet(TStr::Fmt("%s.txt", OutFNm.CStr()));

  // save edge info
  if (ComputeInfo)
    NIMT.SaveEdgeInfo(TStr::Fmt("%s-edge.info", OutFNm.CStr()));

  // save obj+bound info
  if (ComputeBound)
      NIMT.SaveObjInfo(TStr::Fmt("%s-obj", OutFNm.CStr()));

  Catch
  printf("\nrun time: %s (%s)\n", ExeTm.GetTmStr(), TSecTm::GetCurTm().GetTmStr().CStr());
  return 0;
}

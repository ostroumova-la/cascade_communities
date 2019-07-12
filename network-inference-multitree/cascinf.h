#ifndef snap_cascinf_h
#define snap_cascinf_h

#include "Snap.h"

// Hit info (timestamp, candidate parent) about a node in a cascade
class THitInfo {
public:
  TInt NId;
  TIntV Parents;
  TFlt Tm, NodeGain;
public:
  THitInfo(const int& NodeId=-1, const double& HitTime=0) : NId(NodeId), Tm(HitTime), NodeGain(1.0) { }
  THitInfo(TSIn& SIn) : NId(SIn), Parents(SIn), Tm(SIn), NodeGain(SIn) { }
  void Save(TSOut& SOut) const { NId.Save(SOut); Parents.Save(SOut); Tm.Save(SOut); NodeGain.Save(SOut); }
  bool operator < (const THitInfo& Hit) const {
    return Tm < Hit.Tm; }
  bool IsParent(const int& ParentId) { return Parents.IsIn(ParentId); }
};

// Cascade
class TCascade {
public:
  THash<TInt, THitInfo> NIdHitH;
  TFlt CurProb, Alpha, Delta;
  TInt Model;
  double Eps;
public:
  TCascade() : NIdHitH(), CurProb(0), Alpha(1.0), Delta(1.0), Eps(1e-64), Model(0) { }
  TCascade(const double &alpha) : NIdHitH(), CurProb(0), Delta(1.0), Eps(1e-64), Model(0) { Alpha = alpha; }
  TCascade(const double &alpha, const int &model) : NIdHitH(), CurProb(0), Delta(1.0), Eps(1e-64) { Alpha = alpha; Model = model; }
  TCascade(const double &alpha, const double &eps) : NIdHitH(), CurProb(0), Delta(1.0), Model(0) { Alpha = alpha; Eps = eps;}
  TCascade(const double &alpha, const int &model, const double &eps) : NIdHitH(), CurProb(0), Delta(1.0) { Alpha = alpha; Model = model; Eps = eps; }
  TCascade(TSIn& SIn) : NIdHitH(SIn), CurProb(SIn), Alpha(SIn) { }
  void Save(TSOut& SOut) const  { NIdHitH.Save(SOut); CurProb.Save(SOut); Alpha.Save(SOut); }
  void Clr() { NIdHitH.Clr(); CurProb = 0; Alpha = 1.0; }
  int Len() const { return NIdHitH.Len(); }
  int GetNode(const int& i) const { return NIdHitH.GetKey(i); }
  TIntV GetParents(const int NId) const { return NIdHitH.GetDat(NId).Parents; }
  void SetAlpha(const double& alpha) { Alpha = alpha; }
  double GetAlpha() const { return Alpha; }
  void SetDelta(const double& delta) { Delta = delta; }
  double GetDelta() const { return Delta; }
  double GetTm(const int& NId) const { return NIdHitH.GetDat(NId).Tm; }
  void Add(const int& NId, const double& HitTm) { NIdHitH.AddDat(NId, THitInfo(NId, HitTm)); }
  void Del(const int& NId) { NIdHitH.DelKey(NId); }
  bool IsNode(const int& NId) const { return NIdHitH.IsKey(NId); }
  void Sort() { NIdHitH.SortByDat(true); }
  double TransProb(const int& NId1, const int& NId2) const;
  double GetProb(const PNGraph& G);
  void InitProb();
  double UpdateProb(const int& N1, const int& N2, const bool& UpdateProb=false);
  bool IsParent(const int& NId, const int& ParentId) { return NIdHitH.GetDat(NId).IsParent(ParentId); }
};

// Node info (name and number of cascades)
class TNodeInfo {
public:
  TStr Name;
  TInt Vol;
public:
  TNodeInfo() { }
  TNodeInfo(const TStr& NodeNm, const int& Volume) : Name(NodeNm), Vol(Volume) { }
  TNodeInfo(TSIn& SIn) : Name(SIn), Vol(SIn) { }
  void Save(TSOut& SOut) const { Name.Save(SOut); Vol.Save(SOut); }
};

// Edge info (name and number of cascades)
class TEdgeInfo {
public:
  TInt Vol;
  TFlt MarginalGain, MarginalBound, MedianTimeDiff, AverageTimeDiff; // we can skip MarginalBound for efficiency if not explicitly required
public:
  TEdgeInfo() { }
  TEdgeInfo(const int& v,
		    const double& mg,
		    const double& mb,
		    const double& mt,
			const double& at) : Vol(v), MarginalGain(mg), MarginalBound(mb), MedianTimeDiff(mt), AverageTimeDiff(at) { }
  TEdgeInfo(const int& v,
		    const double& mg,
		    const double& mt,
			const double& at) : Vol(v), MarginalGain(mg), MarginalBound(0), MedianTimeDiff(mt), AverageTimeDiff(at) { }
  TEdgeInfo(TSIn& SIn) : Vol(SIn), MarginalGain(SIn), MarginalBound(SIn), MedianTimeDiff(SIn), AverageTimeDiff(SIn) { }
  void Save(TSOut& SOut) const { Vol.Save(SOut); SOut.Save(MarginalGain); SOut.Save(MarginalBound); SOut.Save(MedianTimeDiff); SOut.Save(AverageTimeDiff); } //
};

// NETINF algorithm class
class TNIMT {
public:
  TVec<TCascade> CascV;
  THash<TInt, TNodeInfo> NodeNmH;
  THash<TIntPr, TEdgeInfo> EdgeInfoH;
  TVec<TPair<TFlt, TIntPr> > EdgeGainV;

  THash<TIntPr, TIntV> CascPerEdge; // To implement localized update
  PNGraph Graph, GroundTruth;
  bool BoundOn, CompareGroundTruth;
  TFltPrV PrecisionRecall;
  TFltV Accuracy;

  TIntPrFltH Alphas, Betas;

public:
  TNIMT( ) { BoundOn = false; CompareGroundTruth=false; }
  TNIMT(bool bo, bool cgt) { BoundOn=bo; CompareGroundTruth=cgt; }
  TNIMT(TSIn& SIn) : CascV(SIn), NodeNmH(SIn) { }
  void Save(TSOut& SOut) const { CascV.Save(SOut); NodeNmH.Save(SOut); }

  void LoadCascadesTxt(TSIn& SIn, const int& Model, const double& alpha, const double& delta, const int& NCascades);
  void LoadGroundTruthTxt(TSIn& SIn);

  void AddGroundTruth(PNGraph& gt) { GroundTruth = gt; }
  void GenerateGroundTruth(const int& TNetwork, const int& NNodes, const int& NEdges, const TStr& NetworkParams);
  void SetModels(const double& minalpha, const double& maxalpha, const double& minbeta, const double& maxbeta);

  void AddCasc(const TStr& CascStr, const int& Model=0, const double& alpha=1.0, const double& delta=1.0);
  void AddCasc(const TCascade& Cascade) { CascV.Add(Cascade); }
  void GenCascade(TCascade& C, const int& TModel, TIntPrIntH& EdgesUsed, const double  &delta,
  						   const double& std_waiting_time=0, const double& std_beta=0);
  void GenNoisyCascade(TCascade& Cascade, const int& TModel, TIntPrIntH& EdgesUsed, const double  &delta, const double& std_waiting_time=0, const double& std_beta=0,
		  	  	  	   const double& PercRndNodes=0, const double& PercRndRemoval=0);
  TCascade & GetCasc(int c) { return CascV[c]; }
  int GetCascs() { return CascV.Len(); }

  int GetNodes() { return Graph->GetNodes(); }
  void AddNodeNm(const int& NId, const TNodeInfo& Info) { NodeNmH.AddDat(NId, Info); }
  TStr GetNodeNm(const int& NId) const { return NodeNmH.GetDat(NId).Name; }
  TNodeInfo GetNodeInfo(const int& NId) const { return NodeNmH.GetDat(NId); }
  bool IsNodeNm(const int& NId) const { return NodeNmH.IsKey(NId); }

  void Init();
  double GetAllCascProb(const int& EdgeN1, const int& EdgeN2);
  TIntPr GetBestEdge(double& CurProb, double& LastGain, bool& msort, int &attempts);
  double GetBound(const TIntPr& Edge, double& CurProb);
  void GreedyOpt(const int& MxEdges);

  void SavePajek(const TStr& OutFNm);
  void SavePlaneTextNet(const TStr& OutFNm);
  void SaveEdgeInfo(const TStr& OutFNm);
  void SaveObjInfo(const TStr& OutFNm);

  void SaveGroundTruth(const TStr& OutFNm);
  void SaveCascades(const TStr& OutFNm);
};

#endif

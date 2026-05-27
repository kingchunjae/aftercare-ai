"""
data/generate_data.py  —  27개 시군구, 4유형 균형 포함
A: 소멸위기+부족(4) B: 소멸위기+과잉(9) C: 비위기+부족(4) D: 비위기+균형(10)
"""
import numpy as np, pandas as pd, os

TYPE_COLORS={"A":"#C0392B","B":"#E67E22","C":"#1B4D6B","D":"#27AE60"}
TYPE_LABELS={"A":"위기+공급부족","B":"위기+공급과잉","C":"비위기+공급부족","D":"비위기+균형"}
TOP3_F={
    "A":["돌봄 대기자 증가율","맞벌이 가구 비율","학생수 감소율"],
    "B":["돌봄 이용률 저하","학생수 급감","통폐합 예정 학교 수"],
    "C":["맞벌이 가구 밀도","신도심 학생 집중","방과후 정원 대비 대기 비율"],
    "D":["출생아수 소폭 감소","이용률 안정","지역 인구 유지"],
}

# target_type 을 명시적으로 지정
REGIONS=[
  {"id":"M01","name":"광역시 동구",  "lat":35.148,"lon":126.923,"t":"C"},
  {"id":"M02","name":"광역시 서구",  "lat":35.152,"lon":126.889,"t":"D"},
  {"id":"M03","name":"광역시 남구",  "lat":35.133,"lon":126.902,"t":"C"},
  {"id":"M04","name":"광역시 북구",  "lat":35.175,"lon":126.912,"t":"D"},
  {"id":"M05","name":"광역시 광산구","lat":35.139,"lon":126.793,"t":"C"},
  {"id":"D01","name":"도 A시",      "lat":34.846,"lon":127.354,"t":"A"},
  {"id":"D02","name":"도 B시",      "lat":34.761,"lon":127.662,"t":"D"},
  {"id":"D03","name":"도 C군",      "lat":34.684,"lon":127.536,"t":"A"},
  {"id":"D04","name":"도 D군",      "lat":34.598,"lon":127.489,"t":"B"},
  {"id":"D05","name":"도 E군",      "lat":34.952,"lon":127.591,"t":"A"},
  {"id":"D06","name":"도 F군",      "lat":34.783,"lon":127.412,"t":"B"},
  {"id":"D07","name":"도 G군",      "lat":35.001,"lon":127.483,"t":"B"},
  {"id":"D08","name":"도 H시",      "lat":34.812,"lon":126.462,"t":"D"},
  {"id":"D09","name":"도 I시",      "lat":34.749,"lon":126.388,"t":"D"},
  {"id":"D10","name":"도 J군",      "lat":34.534,"lon":126.347,"t":"A"},
  {"id":"D11","name":"도 K군",      "lat":34.456,"lon":126.297,"t":"B"},
  {"id":"D12","name":"도 L군",      "lat":34.621,"lon":126.189,"t":"B"},
  {"id":"D13","name":"도 M군",      "lat":34.398,"lon":126.094,"t":"B"},
  {"id":"D14","name":"도 N군",      "lat":34.302,"lon":126.501,"t":"B"},
  {"id":"D15","name":"도 O시",      "lat":35.183,"lon":126.982,"t":"D"},
  {"id":"D16","name":"도 P시",      "lat":35.227,"lon":127.142,"t":"D"},
  {"id":"D17","name":"도 Q시",      "lat":35.071,"lon":127.058,"t":"C"},
  {"id":"D18","name":"도 R군",      "lat":35.312,"lon":127.248,"t":"B"},
  {"id":"D19","name":"도 S군",      "lat":35.389,"lon":127.012,"t":"D"},
  {"id":"D20","name":"도 T군",      "lat":35.441,"lon":126.897,"t":"D"},
  {"id":"D21","name":"도 U군",      "lat":35.498,"lon":126.712,"t":"B"},
  {"id":"D22","name":"도 V군",      "lat":35.256,"lon":126.623,"t":"D"},
]

SPECS={
    "A": dict(urban=False,decline=True, dual=(48,60), stu=(60,280),
              sup_ratio=(0.10,0.28), util=(0.70,0.95), wait=(60,200),
              bc=(0.50,0.68), sing=(15,22)),
    "B": dict(urban=False,decline=True, dual=(42,56), stu=(60,350),
              sup_ratio=(0.70,1.20), util=(0.20,0.45), wait=(0,25),
              bc=(0.48,0.70), sing=(13,21)),
    "C": dict(urban=True, decline=False,dual=(64,78), stu=(1200,3500),
              sup_ratio=(0.22,0.40), util=(0.88,0.99), wait=(80,350),
              bc=(0.82,1.05), sing=(5,11)),
    "D": dict(urban=False,decline=False,dual=(55,68), stu=(200,800),
              sup_ratio=(0.50,0.90), util=(0.55,0.80), wait=(5,60),
              bc=(0.80,1.02), sing=(8,14)),
}

rows=[]
for r in REGIONS:
    sp=SPECS[r["t"]]; g=np.random.default_rng(hash(r["id"])%2**31)
    stu=int(g.integers(sp["stu"][0],sp["stu"][1]))
    dual=round(g.uniform(*sp["dual"]),1)
    sing=round(g.uniform(*sp["sing"]),1)
    area=round(g.uniform(20,85),1) if sp["urban"] else round(g.uniform(150,750),1)
    births=int(stu*g.uniform(0.45,0.70))
    births_p=int(births*g.uniform(*sp["bc"]))
    bchg=round((births_p-births)/max(births,1)*100,1)

    tot_cap=int(stu*g.uniform(*sp["sup_ratio"]))
    a_cap=int(tot_cap*g.uniform(0.45,0.65)); c_cap=tot_cap-a_cap
    c_cap=max(c_cap,1)
    c_enr=int(c_cap*g.uniform(*sp["util"]))
    wait=int(g.integers(*sp["wait"]))
    util=round(c_enr/c_cap*100,1)

    dem=round(dual/100,4)
    sup=round(tot_cap/max(stu,1),4); sup=max(sup,0.01)
    imb=round(dem/sup,4)
    dem5=round(dem*(1+bchg/100*0.6),4); dem5=max(dem5,0.01)
    rs=int(max(0,min(100,(imb-1)*20+(20 if sp["decline"] else 0)+sing*0.6+max(0,-bchg)*0.3)))

    rows.append({
        "region_id":r["id"],"name":r["name"],"lat":r["lat"],"lon":r["lon"],
        "urban":sp["urban"],"decline":sp["decline"],
        "students":stu,"dual_income_pct":dual,"single_parent_pct":sing,"area_km2":area,
        "births_5y":births,"births_proj_5y":births_p,"birth_change_pct":bchg,
        "afterschool_cap":a_cap,"care_cap":c_cap,"care_enrolled":c_enr,
        "care_waitlist":wait,"care_util_rate":util,
        "demand_idx":dem,"supply_idx":sup,"imbal_idx":imb,
        "demand_idx_5y":dem5,"region_type":r["t"],
        "type_color":TYPE_COLORS[r["t"]],"type_label":TYPE_LABELS[r["t"]],
        "top3_features":"|".join(TOP3_F[r["t"]]),"risk_score":rs,
    })

df=pd.DataFrame(rows)
out=os.path.join(os.path.dirname(__file__),"regions.csv")
df.to_csv(out,index=False,encoding="utf-8-sig")
print(f"생성 완료: {len(df)}개 / 유형 분포:",df["region_type"].value_counts().to_dict())
print(df[["name","region_type","imbal_idx","risk_score"]].to_string(index=False))

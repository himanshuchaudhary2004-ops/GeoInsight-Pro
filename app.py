import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
from shapely.ops import unary_union

try:
    import folium
    from folium.plugins import HeatMap, MarkerCluster
    from streamlit_folium import st_folium
    MAP_OK = True
except Exception:
    MAP_OK = False

try:
    import rasterio
    RASTER_OK = True
except Exception:
    RASTER_OK = False

try:
    import matplotlib.pyplot as plt
    MPL_OK = True
except Exception:
    MPL_OK = False

try:
    from scipy import stats
    from scipy.spatial import cKDTree
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    from libpysal.weights import KNN
    from esda.moran import Moran, Moran_Local
    from esda.getisord import G_Local
    PYSAL_OK = True
except Exception:
    PYSAL_OK = False

st.set_page_config(page_title="GeoInsight Pro 50", page_icon="🌍", layout="wide")

st.markdown("""
<style>
.block-container{max-width:1600px;padding-top:1rem}
.hero{padding:32px 38px;border-radius:26px;color:#fff;
background:linear-gradient(135deg,#020617,#123b72 55%,#0e7490);
box-shadow:0 16px 45px rgba(2,8,23,.22);margin-bottom:18px}
.hero h1{margin:0;font-size:2.65rem}.hero p{opacity:.9}
.badge{display:inline-block;background:#ffffff22;border-radius:999px;padding:5px 10px;margin:10px 5px 0 0;font-size:.74rem}
div[data-testid="stMetric"]{border:1px solid #e2e8f0;border-radius:14px;padding:10px}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🌍 GeoInsight Pro 50</h1>
<p>Professional Geoinformatics workbench for GIS, Remote Sensing, Spatial Statistics, AI/ML, Terrain, Accessibility and Decision Support.</p>
<span class="badge">Vector</span><span class="badge">Raster</span><span class="badge">Remote Sensing</span>
<span class="badge">Spatial Statistics</span><span class="badge">AI/ML</span><span class="badge">Hydrology</span>
<span class="badge">Network</span><span class="badge">MCDA</span><span class="badge">Cartography</span>
</div>
""", unsafe_allow_html=True)

for k,v in {
    "layers":{}, "active":None, "result":None, "raster":None,
    "raster_result":None, "history":[], "project":"GeoInsight Project",
    "model":None, "report_notes":[]
}.items():
    if k not in st.session_state: st.session_state[k]=v

def log(x):
    st.session_state.history=(st.session_state.history+[x])[-40:]

def current():
    return st.session_state.layers.get(st.session_state.active)

def numcols(g):
    return [c for c in g.columns if c!="geometry" and pd.api.types.is_numeric_dtype(g[c])]

def load_vector(up):
    suf=Path(up.name).suffix.lower(); raw=up.getvalue()
    if suf in (".geojson",".json"): return gpd.read_file(io.BytesIO(raw))
    if suf==".csv":
        df=pd.read_csv(io.BytesIO(raw)); low={c.lower():c for c in df.columns}
        lat=next((low[x] for x in ("lat","latitude","y") if x in low),None)
        lon=next((low[x] for x in ("lon","longitude","lng","x") if x in low),None)
        if not lat or not lon: raise ValueError("CSV needs latitude/longitude or x/y columns.")
        return gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df[lon],df[lat]),crs=4326)
    if suf in (".zip",".gpkg",".parquet"):
        with tempfile.NamedTemporaryFile(suffix=suf,delete=False) as f:
            f.write(raw); path=f.name
        try:
            return gpd.read_parquet(path) if suf==".parquet" else gpd.read_file("zip://"+path if suf==".zip" else path)
        finally:
            try: os.unlink(path)
            except OSError: pass
    raise ValueError("Unsupported vector format.")

def add_layer(name,g):
    base=name; name=base; i=2
    while name in st.session_state.layers:
        name=f"{base}_{i}"; i+=1
    st.session_state.layers[name]=g; st.session_state.active=name; log("Loaded "+name)

def metric_crs(g):
    if g.crs is None: raise ValueError("CRS is undefined.")
    return g.to_crs(g.estimate_utm_crs()) if getattr(g.crs,"is_geographic",False) else g

def web(g):
    return g.set_crs(4326,allow_override=True) if g.crs is None else g.to_crs(4326)

def show_map(g,mode="feature",field=None):
    if not MAP_OK: st.info("Install folium and streamlit-folium for interactive maps."); return
    if g is None or g.empty: return
    w=web(g); c=w.geometry.unary_union.centroid
    m=folium.Map([c.y,c.x],zoom_start=6,tiles="CartoDB positron",control_scale=True)
    if mode=="heat" and (w.geometry.geom_type=="Point").any():
        pts=w[w.geometry.geom_type=="Point"]; data=[]
        for _,r in pts.iterrows():
            data.append([r.geometry.y,r.geometry.x,float(r[field]) if field and pd.notna(r[field]) else 1])
        HeatMap(data,radius=15,blur=18).add_to(m)
    elif mode=="cluster" and (w.geometry.geom_type=="Point").any():
        mc=MarkerCluster().add_to(m)
        for _,r in w.iterrows(): folium.Marker([r.geometry.y,r.geometry.x]).add_to(mc)
    else:
        fields=[c for c in w.columns if c!="geometry"][:8]
        folium.GeoJson(w.to_json(),tooltip=folium.GeoJsonTooltip(fields=fields) if fields else None).add_to(m)
    st_folium(m,height=600,returned_objects=[])

def manifest():
    return json.dumps({"project":st.session_state.project,"layers":list(st.session_state.layers),
                       "active":st.session_state.active,"history":st.session_state.history},indent=2).encode()

# Sidebar
with st.sidebar:
    st.markdown("## 🧭 GeoInsight Pro 50")
    st.caption("50-capability professional student workbench")
    st.session_state.project=st.text_input("Project name",st.session_state.project)
    workspace=st.radio("Workspace",[
        "🏠 Command Center","📁 Data & Layers","🗺️ Cartography","🧪 QA & CRS",
        "📐 Geoprocessing","📏 Distance & Proximity","📊 Spatial Statistics",
        "🔥 Hotspot Analysis","🎯 MCDA / Suitability","🛰️ Remote Sensing",
        "🌱 LULC Classification","🤖 Machine Learning","⛰️ DEM & Terrain",
        "💧 Hydrology","🧮 Raster Calculator","📈 Change Detection",
        "📍 Interpolation","🌐 Accessibility","🕸️ Network Concepts",
        "📋 Sampling & Data Prep","📄 Report & Export","🎓 GIS Academy"
    ])
    st.divider()
    st.caption("Data → QA → CRS → Analysis → Validation → Report")
    if st.session_state.active: st.caption("Active: "+st.session_state.active)

# 1
if workspace=="🏠 Command Center":
    st.subheader("Command Center")
    if not st.session_state.layers: st.info("Load a dataset in Data & Layers.")
    else:
        g=current(); a,b,c,d,e=st.columns(5)
        a.metric("Layers",len(st.session_state.layers)); b.metric("Features",len(g))
        c.metric("Attributes",len(g.columns)-1); d.metric("CRS",str(g.crs)); e.metric("Operations",len(st.session_state.history))
        st.dataframe(g.drop(columns="geometry").head(15),use_container_width=True)
        show_map(g)
        st.write({"invalid geometries":int((~g.geometry.is_valid).sum()),
                  "empty geometries":int(g.geometry.is_empty.sum()),
                  "missing attribute cells":int(g.drop(columns="geometry").isna().sum().sum())})

# 2
elif workspace=="📁 Data & Layers":
    st.subheader("📁 Data & Layer Manager")
    ups=st.file_uploader("Vector files",type=["geojson","json","csv","zip","gpkg","parquet"],accept_multiple_files=True)
    if ups:
        for up in ups:
            try: add_layer(Path(up.name).stem,load_vector(up)); st.success("Loaded "+up.name)
            except Exception as e: st.error(f"{up.name}: {e}")
    tif=st.file_uploader("Raster GeoTIFF",type=["tif","tiff"],accept_multiple_files=False)
    if tif and RASTER_OK:
        st.session_state.raster={"raw":tif.getvalue()}
        with rasterio.MemoryFile(tif.getvalue()) as mem:
            with mem.open() as src:
                st.session_state.raster["profile"]=src.profile
                st.session_state.raster["arr"]=src.read(1)
        st.success("Raster loaded.")
    if st.session_state.layers:
        names=list(st.session_state.layers)
        st.session_state.active=st.selectbox("Active layer",names,index=names.index(st.session_state.active) if st.session_state.active in names else 0)
        for n in names:
            g=st.session_state.layers[n]
            with st.expander(("🟢 " if n==st.session_state.active else "⚪ ")+n):
                st.write({"features":len(g),"CRS":str(g.crs),"geometry":g.geometry.geom_type.mode().iloc[0] if len(g) else "—"})
        st.download_button("💾 Project manifest",manifest(),st.session_state.project+".json","application/json")

# 3
elif workspace=="🗺️ Cartography":
    st.subheader("🗺️ Cartography Studio")
    g=current()
    if g is None: st.warning("Load a layer.")
    else:
        a,b=st.columns([1,2])
        with a:
            mode=st.selectbox("Map type",["feature","heat","cluster"])
            field=st.selectbox("Weight field",["None"]+numcols(g))
            st.markdown("**Map QA checklist**")
            st.write("Title • legend • scale • source • CRS • units • classification • uncertainty")
        with b: show_map(g,mode,None if field=="None" else field)
        st.dataframe(g.drop(columns="geometry").head(20),use_container_width=True)

# 4
elif workspace=="🧪 QA & CRS":
    st.subheader("🧪 Data Quality, Geometry & CRS")
    g=current()
    if g is None: st.warning("Load a layer.")
    else:
        t1,t2,t3=st.tabs(["Profile","Geometry","CRS"])
        with t1:
            st.dataframe(g.dtypes.astype(str).rename("dtype")); st.write("Bounds:",g.total_bounds)
            st.write("Missing:",g.drop(columns="geometry").isna().sum())
        with t2:
            st.metric("Invalid",int((~g.geometry.is_valid).sum()))
            if st.button("Repair geometry"):
                r=g.copy(); r["geometry"]=r.geometry.make_valid(); add_layer("Repaired",r)
        with t3:
            st.write("CRS:",g.crs)
            if st.button("Suggest UTM"):
                try: st.success(str(g.estimate_utm_crs()))
                except Exception as e: st.error(str(e))

# 5
elif workspace=="📐 Geoprocessing":
    st.subheader("📐 Geoprocessing Toolbox")
    g=current()
    if g is None: st.warning("Load a layer.")
    else:
        op=st.selectbox("Operation",["Reproject","Buffer","Dissolve","Centroid","Convex Hull","Explode","Clip","Intersection","Union","Difference","Spatial Join","Attribute Query"])
        if op=="Reproject":
            epsg=st.number_input("EPSG",1,999999,4326)
            if st.button("Run"): st.session_state.result=g.to_crs(int(epsg)); log("Reproject")
        elif op=="Buffer":
            dist=st.number_input("Distance",0.,1e9,1000.)
            if st.button("Run"):
                try:
                    p=metric_crs(g); r=p.copy(); r["geometry"]=r.geometry.buffer(dist); st.session_state.result=r; log("Buffer")
                except Exception as e: st.error(str(e))
        elif op=="Dissolve":
            f=st.selectbox("Field",["All"]+[c for c in g.columns if c!="geometry"])
            if st.button("Run"):
                st.session_state.result=gpd.GeoDataFrame({"id":[1]},geometry=[unary_union(g.geometry)],crs=g.crs) if f=="All" else g.dissolve(by=f,as_index=False)
                log("Dissolve")
        elif op=="Centroid":
            if st.button("Run"):
                p=metric_crs(g); r=p.copy(); r["geometry"]=r.geometry.centroid; st.session_state.result=r; log("Centroid")
        elif op=="Convex Hull":
            if st.button("Run"):
                st.session_state.result=gpd.GeoDataFrame({"id":[1]},geometry=[unary_union(g.geometry).convex_hull],crs=g.crs); log("Convex Hull")
        elif op=="Explode":
            if st.button("Run"): st.session_state.result=g.explode(index_parts=True).reset_index(drop=True); log("Explode")
        elif op=="Attribute Query":
            f=st.selectbox("Field",[c for c in g.columns if c!="geometry"]); value=st.text_input("Exact value")
            if st.button("Run"): st.session_state.result=g[g[f].astype(str)==value].copy(); log("Attribute Query")
        else:
            up=st.file_uploader("Second layer",type=["geojson","json","zip","gpkg","parquet"],key="overlay")
            if up:
                other=load_vector(up)
                if other.crs!=g.crs: other=other.to_crs(g.crs)
                if st.button("Run"):
                    if op=="Spatial Join": r=gpd.sjoin(g,other,how="left",predicate="intersects")
                    else: r=gpd.overlay(g,other,how={"Clip":"intersection","Intersection":"intersection","Union":"union","Difference":"difference"}[op])
                    st.session_state.result=r; log(op)
        if st.session_state.result is not None: show_map(st.session_state.result)

# 6
elif workspace=="📏 Distance & Proximity":
    st.subheader("📏 Distance & Proximity")
    g=current()
    if g is None: st.warning("Load a layer.")
    else:
        tool=st.selectbox("Tool",["Nearest feature","Distance matrix","Service buffers","Coordinates"])
        if tool=="Coordinates":
            r=web(g).copy(); r["longitude"]=r.geometry.x; r["latitude"]=r.geometry.y; st.dataframe(r.drop(columns="geometry"),use_container_width=True)
        elif tool=="Service buffers":
            d=st.number_input("Radius",1.,1e8,1000.)
            if st.button("Run"):
                p=metric_crs(g); r=p.copy(); r["geometry"]=r.geometry.buffer(d); st.session_state.result=r; show_map(r)
        elif tool=="Nearest feature":
            up=st.file_uploader("Target",type=["geojson","json","zip","gpkg"],key="nearest")
            if up:
                o=load_vector(up)
                if o.crs!=g.crs:o=o.to_crs(g.crs)
                if st.button("Run"):
                    p=metric_crs(g); q=metric_crs(o); ids=[]; ds=[]
                    for geom in p.geometry:
                        d=q.geometry.distance(geom); i=d.idxmin(); ids.append(i); ds.append(float(d.loc[i]))
                    r=p.copy(); r["nearest_id"]=ids; r["distance"]=ds; st.session_state.result=r
        else:
            if st.button("Calculate"):
                p=metric_crs(g); geoms=p.geometry.tolist()
                st.dataframe(pd.DataFrame([[float(a.distance(b)) for b in geoms] for a in geoms]),use_container_width=True)

# 7
elif workspace=="📊 Spatial Statistics":
    st.subheader("📊 Spatial Statistics")
    g=current()
    if g is None: st.warning("Load a layer.")
    else:
        tabs=st.tabs(["Descriptive","Moran I","Local Moran","Getis-Ord Gi*","Trend"])
        with tabs[0]: st.dataframe(g.describe(include="all").T,use_container_width=True)
        with tabs[1]:
            fs=numcols(g)
            if PYSAL_OK and len(g)>=5 and fs:
                f=st.selectbox("Variable",fs,key="mi"); p=metric_crs(g).reset_index(drop=True); w=KNN.from_dataframe(p,k=min(8,len(p)-1)); w.transform="r"
                x=pd.to_numeric(p[f],errors="coerce").fillna(pd.to_numeric(p[f],errors="coerce").mean()); m=Moran(x,w)
                st.metric("Moran I",f"{m.I:.5f}"); st.metric("p (permutation)",f"{m.p_sim:.5f}")
            else: st.info("Install PySAL and use >=5 features.")
        with tabs[2]:
            fs=numcols(g)
            if PYSAL_OK and fs:
                f=st.selectbox("Variable",fs,key="lm"); p=metric_crs(g).reset_index(drop=True); w=KNN.from_dataframe(p,k=min(8,len(p)-1)); w.transform="r"; x=pd.to_numeric(p[f],errors="coerce").fillna(pd.to_numeric(p[f],errors="coerce").mean()).to_numpy(); m=Moran_Local(x,w,permutations=999)
                r=p.copy(); r["local_I"]=m.Is; r["p_sim"]=m.p_sim; st.session_state.result=r; st.dataframe(r.drop(columns="geometry").head(30))
        with tabs[3]:
            fs=numcols(g)
            if PYSAL_OK and fs:
                f=st.selectbox("Variable",fs,key="gi"); p=metric_crs(g).reset_index(drop=True); w=KNN.from_dataframe(p,k=min(8,len(p)-1)); w.transform="r"; x=pd.to_numeric(p[f],errors="coerce").fillna(pd.to_numeric(p[f],errors="coerce").mean()).to_numpy(); gi=G_Local(x,w,permutations=999)
                r=p.copy(); r["Gi_Z"]=gi.Zs; r["p_sim"]=gi.p_sim; r["significance"]=np.select([r.p_sim<.01,r.p_sim<.05,r.p_sim<.1],["99%","95%","90%"],default="NS"); st.session_state.result=r; st.dataframe(r.drop(columns="geometry").head(30))
        with tabs[4]:
            fs=numcols(g)
            if SCIPY_OK and fs:
                f=st.selectbox("Variable",fs,key="trend"); x=pd.to_numeric(g[f],errors="coerce"); ok=x.notna(); rho,p=stats.spearmanr(np.arange(len(x))[ok],x[ok]); st.metric("Spearman rho",f"{rho:.4f}"); st.metric("p-value",f"{p:.5f}")

# 8
elif workspace=="🔥 Hotspot Analysis":
    st.subheader("🔥 Hotspot / Density Explorer")
    g=current()
    if g is None: st.warning("Load point data.")
    else:
        pts=g[g.geometry.geom_type=="Point"]
        if len(pts)<2: st.info("Need at least two points.")
        else:
            method=st.selectbox("Method",["Heatmap","Grid density","Z-score"])
            if method=="Heatmap": show_map(pts,"heat")
            elif method=="Grid density":
                p=metric_crs(pts); cell=st.number_input("Cell size",1.,1e8,1000.); minx,miny,_,_=p.total_bounds; r=p.copy(); r["gx"]=np.floor((r.geometry.x-minx)/cell); r["gy"]=np.floor((r.geometry.y-miny)/cell); counts=r.groupby(["gx","gy"]).size().rename("count").reset_index(); st.dataframe(counts,use_container_width=True)
            else:
                fs=numcols(pts)
                if fs:
                    f=st.selectbox("Variable",fs); x=pd.to_numeric(pts[f],errors="coerce").fillna(0); sd=x.std(ddof=0); r=pts.copy(); r["z"]=(x-x.mean())/sd if sd else 0; st.session_state.result=r; show_map(r)

# 9
elif workspace=="🎯 MCDA / Suitability":
    st.subheader("🎯 Multi-Criteria Decision Analysis")
    g=current()
    if g is None: st.warning("Load a layer.")
    else:
        fs=numcols(g); selected=st.multiselect("Criteria",fs,default=fs[:min(6,len(fs))]); config=[]
        for f in selected:
            a,b,c=st.columns(3)
            with a: direction=st.selectbox(f,["Higher is better","Lower is better"],key="md_"+f)
            with b: weight=st.number_input("Weight",0.,100.,100/len(selected),key="mw_"+f)
            with c: scale=st.selectbox("Scale",["Min-Max","Z-score"],key="ms_"+f)
            config.append((f,direction,weight,scale))
        if selected and st.button("Run suitability"):
            total=sum(x[2] for x in config); r=g.copy(); score=pd.Series(0.,index=r.index)
            for f,d,w,s in config:
                x=pd.to_numeric(r[f],errors="coerce")
                if s=="Min-Max":
                    lo,hi=x.min(),x.max(); z=(x-lo)/(hi-lo) if hi!=lo else x*0+1
                else:
                    sd=x.std(ddof=0); z=(x-x.mean())/sd if sd else x*0; rng=z.max()-z.min(); z=(z-z.min())/rng if rng else z*0+1
                if d=="Lower is better": z=1-z
                score+=z.fillna(0)*w/total
            r["score"]=score; r["class"]=pd.cut(score,[-.01,.2,.4,.6,.8,1.01],labels=["Very Low","Low","Moderate","High","Very High"]); st.session_state.result=r; st.dataframe(r.drop(columns="geometry").sort_values("score",ascending=False).head(30))

# 10
elif workspace=="🛰️ Remote Sensing":
    st.subheader("🛰️ Sentinel / Landsat Spectral Analysis")
    r=st.session_state.raster
    if r is None: st.info("Load a GeoTIFF in Data & Layers.")
    else:
        with rasterio.MemoryFile(r["raw"]) as mem:
            with mem.open() as src:
                n=src.count; product=st.selectbox("Product",["Generic","Sentinel-2","Landsat 8/9","Landsat 5/7"])
                mode=st.selectbox("Analysis",["Normalized Difference Index","Band Statistics","Composite Preview"])
                if mode=="Normalized Difference Index":
                    idx=st.selectbox("Index",["NDVI","NDWI","NDBI","NDSI","Custom"])
                    a=st.number_input("Band A",1,n,1); b=st.number_input("Band B",1,n,min(2,n))
                    if st.button("Calculate"):
                        A=src.read(a).astype(float); B=src.read(b).astype(float); out=np.where(np.abs(A+B)<1e-9,np.nan,(A-B)/(A+B)); st.session_state.raster_result=out
                        if MPL_OK:
                            fig,ax=plt.subplots(figsize=(10,5)); im=ax.imshow(out,cmap="RdYlGn",vmin=-1,vmax=1); fig.colorbar(im,ax=ax); ax.axis("off"); ax.set_title(idx); st.pyplot(fig); plt.close(fig)
                elif mode=="Band Statistics":
                    rows=[]
                    for i in range(1,n+1):
                        x=src.read(i).astype(float); rows.append([i,np.nanmin(x),np.nanmax(x),np.nanmean(x),np.nanstd(x)])
                    st.dataframe(pd.DataFrame(rows,columns=["Band","Min","Max","Mean","Std"]),use_container_width=True)
                else:
                    bands=st.multiselect("Bands",list(range(1,n+1)),default=list(range(1,min(3,n)+1)))
                    if bands:
                        cube=np.stack([src.read(b).astype(float) for b in bands],axis=-1); cube=(cube-np.nanmin(cube,axis=(0,1)))/(np.nanmax(cube,axis=(0,1))-np.nanmin(cube,axis=(0,1))+1e-9); st.image(cube,caption="Normalized composite")
        st.caption("Always verify sensor metadata, band definitions and QA masks before scientific interpretation.")

# 11
elif workspace=="🌱 LULC Classification":
    st.subheader("🌱 LULC Classification")
    r=st.session_state.raster
    if r is None: st.info("Load multiband GeoTIFF.")
    elif not SKLEARN_OK: st.warning("Install scikit-learn.")
    else:
        with rasterio.MemoryFile(r["raw"]) as mem:
            with mem.open() as src:
                n=src.count; bands=st.multiselect("Feature bands",list(range(1,n+1)),default=list(range(1,min(5,n)+1))); k=st.number_input("Classes",2,20,5)
                if bands and st.button("Run K-Means"):
                    cube=np.stack([src.read(b).astype(float) for b in bands],axis=-1); flat=cube.reshape(-1,len(bands)); valid=np.isfinite(flat).all(axis=1); X=flat[valid]
                    km=KMeans(n_clusters=k,n_init=10,random_state=42); labels=km.fit_predict(X)+1; out=np.zeros(flat.shape[0],np.uint8); out[valid]=labels; out=out.reshape(cube.shape[:2]); st.session_state.raster_result=out
                    if MPL_OK:
                        fig,ax=plt.subplots(figsize=(10,5)); ax.imshow(out,cmap="tab20"); ax.axis("off"); ax.set_title("Exploratory LULC classes"); st.pyplot(fig); plt.close(fig)
                    st.metric("Classified pixels",int(valid.sum()))
                st.info("For dissertation use, add labelled training samples and independent validation/confusion-matrix assessment.")

# 12
elif workspace=="🤖 Machine Learning":
    st.subheader("🤖 Machine Learning Lab")
    g=current()
    if g is None: st.warning("Load vector data.")
    elif not SKLEARN_OK: st.warning("Install scikit-learn.")
    else:
        df=g.drop(columns="geometry"); target=st.selectbox("Target",list(df.columns)); feats=st.multiselect("Numeric features",[c for c in df.columns if c!=target and pd.api.types.is_numeric_dtype(df[c])]); task=st.selectbox("Task",["Classification","Regression"]); model_name=st.selectbox("Model",["Random Forest","Linear/Logistic"])
        if feats and st.button("Train"):
            d=df[feats+[target]].dropna()
            if len(d)<20: st.error("Need at least 20 complete observations.")
            else:
                X=d[feats].to_numpy(); y=d[target].to_numpy(); strat=y if task=="Classification" and len(np.unique(y))>1 else None
                Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=strat)
                if task=="Classification":
                    model=RandomForestClassifier(n_estimators=250,random_state=42,class_weight="balanced") if model_name=="Random Forest" else make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))
                    model.fit(Xtr,ytr); pred=model.predict(Xte); st.metric("Accuracy",f"{accuracy_score(yte,pred):.4f}"); st.dataframe(pd.DataFrame(confusion_matrix(yte,pred)))
                else:
                    model=RandomForestRegressor(n_estimators=250,random_state=42) if model_name=="Random Forest" else make_pipeline(StandardScaler(),Ridge())
                    model.fit(Xtr,ytr); pred=model.predict(Xte); st.metric("RMSE",f"{mean_squared_error(yte,pred)**.5:.4f}"); st.metric("R²",f"{r2_score(yte,pred):.4f}")
                st.session_state.model=model; log("ML training")
                if hasattr(model,"feature_importances_"): st.dataframe(pd.DataFrame({"feature":feats,"importance":model.feature_importances_}).sort_values("importance",ascending=False))

# 13
elif workspace=="⛰️ DEM & Terrain":
    st.subheader("⛰️ DEM & Terrain Analysis")
    r=st.session_state.raster
    if r is None: st.info("Load DEM GeoTIFF.")
    else:
        a=np.asarray(r["arr"],float); tr=r["profile"]["transform"]; gy,gx=np.gradient(a,abs(tr.e),abs(tr.a)); tool=st.selectbox("Terrain product",["Elevation","Slope","Aspect","Hillshade","Ruggedness","Curvature"])
        if tool=="Elevation": out=a
        elif tool=="Slope": out=np.degrees(np.arctan(np.sqrt(gx*gx+gy*gy)))
        elif tool=="Aspect": out=(np.degrees(np.arctan2(-gx,gy))+360)%360
        elif tool=="Hillshade":
            sl=np.pi/2-np.arctan(np.sqrt(gx*gx+gy*gy)); asp=np.arctan2(-gx,gy); out=255*(np.cos(np.radians(45))*np.cos(sl)+np.sin(np.radians(45))*np.sin(sl)*np.cos(np.radians(315)-asp))
        elif tool=="Ruggedness": out=np.sqrt(gx*gx+gy*gy)
        else:
            gyy,gxx=np.gradient(gy,abs(tr.e),abs(tr.a)); out=gxx+gyy
        st.session_state.raster_result=out
        if MPL_OK:
            fig,ax=plt.subplots(figsize=(10,5)); im=ax.imshow(out,cmap="terrain"); fig.colorbar(im,ax=ax); ax.axis("off"); ax.set_title(tool); st.pyplot(fig); plt.close(fig)

# 14
elif workspace=="💧 Hydrology":
    st.subheader("💧 Hydrology Preview")
    r=st.session_state.raster
    if r is None: st.info("Load DEM.")
    else:
        a=np.asarray(r["arr"],float); gy,gx=np.gradient(a); slope=np.sqrt(gx*gx+gy*gy)
        tool=st.selectbox("Product",["Low-slope drainage candidate","Relative elevation","Flow direction preview"])
        if tool=="Low-slope drainage candidate": out=(slope<=np.nanpercentile(slope,20)).astype(np.uint8)
        elif tool=="Relative elevation": out=(a-np.nanmin(a))/(np.nanmax(a)-np.nanmin(a)+1e-9)
        else: out=np.arctan2(-gy,-gx)
        if MPL_OK:
            fig,ax=plt.subplots(figsize=(10,5)); ax.imshow(out,cmap="Blues"); ax.axis("off"); ax.set_title(tool); st.pyplot(fig); plt.close(fig)
        st.caption("This is a terrain preview, not a full hydrological model. Rigorous watershed work requires sink filling, flow routing and accumulation validation.")

# 15
elif workspace=="🧮 Raster Calculator":
    st.subheader("🧮 Raster Calculator & Reclassification")
    r=st.session_state.raster
    if r is None: st.info("Load raster.")
    else:
        with rasterio.MemoryFile(r["raw"]) as mem:
            with mem.open() as src:
                tabs=st.tabs(["Calculator","Reclassify","Statistics"])
                with tabs[0]:
                    expr=st.text_input("Expression","B1")
                    if st.button("Evaluate"):
                        env={f"B{i}":src.read(i).astype(float) for i in range(1,src.count+1)}
                        if all(x in "0123456789B +-*/()." for x in expr):
                            try: st.session_state.raster_result=eval(expr,{"__builtins__":{}},env); st.success("Calculated.")
                            except Exception as e: st.error(str(e))
                        else: st.error("Only B1/B2/... arithmetic expressions are allowed.")
                with tabs[1]:
                    b=st.number_input("Band",1,src.count,1); lo=st.number_input("Min",value=0.); hi=st.number_input("Max",value=1.); cls=st.number_input("New class",value=1.)
                    if st.button("Reclassify"): st.session_state.raster_result=np.where((src.read(b)>=lo)&(src.read(b)<=hi),cls,src.read(b))
                with tabs[2]:
                    rows=[]
                    for i in range(1,src.count+1):
                        x=src.read(i).astype(float); rows.append([i,np.nanmin(x),np.nanmax(x),np.nanmean(x),np.nanstd(x)])
                    st.dataframe(pd.DataFrame(rows,columns=["Band","Min","Max","Mean","Std"]))

# 16
elif workspace=="📈 Change Detection":
    st.subheader("📈 Change Detection")
    a=st.file_uploader("Time 1 GeoTIFF",type=["tif","tiff"],key="ca"); b=st.file_uploader("Time 2 GeoTIFF",type=["tif","tiff"],key="cb")
    if a and b:
        with rasterio.MemoryFile(a.getvalue()) as ma,rasterio.MemoryFile(b.getvalue()) as mb:
            with ma.open() as A,mb.open() as B:
                if (A.width,A.height,A.crs)!=(B.width,B.height,B.crs): st.error("Matching grid dimensions and CRS required.")
                else:
                    x=A.read(1).astype(float); y=B.read(1).astype(float); m=st.selectbox("Method",["Difference","Relative %","Normalized difference"])
                    out=y-x if m=="Difference" else np.where(np.abs(x)<1e-9,np.nan,(y-x)/x*100) if m=="Relative %" else np.where(np.abs(x+y)<1e-9,np.nan,(y-x)/(y+x))
                    st.metric("Mean change",f"{float(np.nanmean(out)):.5f}")
                    if MPL_OK:
                        fig,ax=plt.subplots(figsize=(10,5)); im=ax.imshow(out,cmap="RdBu_r"); fig.colorbar(im,ax=ax); ax.axis("off"); ax.set_title(m); st.pyplot(fig); plt.close(fig)

# 17
elif workspace=="📍 Interpolation":
    st.subheader("📍 Spatial Interpolation")
    g=current()
    if g is None: st.warning("Load point observations.")
    else:
        p=g[g.geometry.geom_type=="Point"]; fs=numcols(p)
        if len(p)<5 or not fs: st.info("Need >=5 points and numeric observations.")
        else:
            f=st.selectbox("Variable",fs); method=st.selectbox("Method",["IDW","Nearest neighbour"]); cell=st.number_input("Cell size",1.,1e7,1000.)
            if st.button("Interpolate"):
                q=metric_crs(p); minx,miny,maxx,maxy=q.total_bounds; xs=np.arange(minx,maxx+cell,cell); ys=np.arange(miny,maxy+cell,cell); xx,yy=np.meshgrid(xs,ys); xy=np.c_[q.geometry.x,q.geometry.y]; z=pd.to_numeric(q[f],errors="coerce").to_numpy(); ok=np.isfinite(z); xy=xy[ok]; z=z[ok]; out=[]
                for x,y in np.c_[xx.ravel(),yy.ravel()]:
                    d=np.sqrt(((xy-[x,y])**2).sum(axis=1))
                    out.append(z[np.argmin(d)] if method=="Nearest neighbour" else np.sum((1/(d*d+1e-12))*z)/np.sum(1/(d*d+1e-12)))
                surf=np.array(out).reshape(xx.shape)
                if MPL_OK:
                    fig,ax=plt.subplots(figsize=(10,6)); im=ax.imshow(surf,extent=[minx,maxx,miny,maxy],origin="lower"); fig.colorbar(im,ax=ax); ax.set_title(method); st.pyplot(fig); plt.close(fig)

# 18
elif workspace=="🌐 Accessibility":
    st.subheader("🌐 Facility Accessibility")
    g=current()
    if g is None: st.warning("Load facilities/users.")
    else:
        up=st.file_uploader("Facility layer",type=["geojson","json","zip","gpkg"],key="acc")
        if up:
            f=load_vector(up)
            if f.crs!=g.crs:f=f.to_crs(g.crs)
            radius=st.number_input("Coverage radius",1.,1e8,2000.)
            if st.button("Calculate coverage"):
                q=metric_crs(g); fac=metric_crs(f); buffers=fac.geometry.buffer(radius); union=unary_union(buffers); r=q.copy(); r["covered"]=r.geometry.intersects(union); st.session_state.result=r; st.metric("Covered",int(r["covered"].sum())); show_map(r)

# 19
elif workspace=="🕸️ Network Concepts":
    st.subheader("🕸️ Network GIS")
    st.write("A true network analysis model needs a graph: **nodes + edges + impedance/cost**.")
    st.info("Use a road/transport graph for shortest path, fastest route, service area, OD matrix and centrality. Straight-line distance is not a network route.")
    st.markdown("### Recommended network workflow")
    st.write("Clean network → snap points → build graph → define impedance → shortest path → accessibility → validate.")

# 20
elif workspace=="📋 Sampling & Data Prep":
    st.subheader("📋 Sampling & Data Preparation")
    g=current()
    if g is None: st.warning("Load a dataset.")
    else:
        method=st.selectbox("Sampling",["Random sample","Systematic row sample","Stratified by field"])
        n=st.number_input("Sample size",1,len(g),min(30,len(g)))
        if method=="Stratified by field":
            f=st.selectbox("Stratification field",[c for c in g.columns if c!="geometry"])
        if st.button("Generate sample"):
            if method=="Random sample": r=g.sample(n=min(n,len(g)),random_state=42)
            elif method=="Systematic row sample": r=g.iloc[np.linspace(0,len(g)-1,n).astype(int)].copy()
            else:
                r=g.groupby(f,group_keys=False).apply(lambda x:x.sample(min(len(x),max(1,int(n*len(x)/len(g)))),random_state=42)).reset_index(drop=True)
            st.session_state.result=r; st.dataframe(r.drop(columns="geometry"),use_container_width=True); show_map(r)

# 21
elif workspace=="📄 Report & Export":
    st.subheader("📄 Report & Export")
    g=st.session_state.result if st.session_state.result is not None else current()
    if g is None: st.info("Load or analyse a vector layer.")
    else:
        st.write({"project":st.session_state.project,"active":st.session_state.active,"features":len(g),"CRS":str(g.crs),"bounds":list(map(float,g.total_bounds))})
        st.download_button("⬇️ GeoJSON",g.to_json().encode(),"geoinsight_result.geojson","application/geo+json")
        st.download_button("⬇️ CSV",g.drop(columns="geometry").to_csv(index=False).encode(),"geoinsight_attributes.csv","text/csv")
        notes=st.text_area("Research notes","")
        report=f"""# {st.session_state.project}
## GeoInsight Pro 50 Analysis Report
Active layer: {st.session_state.active}
Features: {len(g)}
CRS: {g.crs}
Bounds: {list(map(float,g.total_bounds))}

## Notes
{notes}

## Analysis history
{chr(10).join("- "+x for x in st.session_state.history)}

## Scientific checklist
- Verify data source and licensing
- Verify CRS and units
- Document preprocessing
- Validate analytical assumptions
- Report uncertainty/limitations
- Preserve reproducibility information
"""
        st.download_button("📄 Markdown report",report.encode(),"geoinsight_report.md","text/markdown")

# 22
else:
    st.subheader("🎓 GIS Academy")
    lessons={
        "CRS":"Use projected CRS for metric operations and document the CRS used.",
        "Overlay":"Clip, intersection, union and difference have different spatial semantics and attribute consequences.",
        "Remote sensing":"Sensor-specific band metadata and QA masking must be checked before calculating spectral indices.",
        "LULC":"Classification requires independent validation and a confusion matrix for defensible accuracy reporting.",
        "Moran's I":"Spatial autocorrelation depends on the weights matrix; report weights and significance testing.",
        "Gi*":"A hotspot map is not automatically statistically significant. Inspect z-scores and p-values.",
        "MCDA":"Document criteria, normalization, weights, preference direction and sensitivity.",
        "DEM hydrology":"Watershed modelling needs validated sink treatment, flow routing and accumulation.",
        "Interpolation":"IDW is a modelling assumption; use cross-validation and inspect residuals.",
        "Machine learning":"Avoid leakage and report an independent validation metric.",
        "Network GIS":"Network distance differs from Euclidean distance because routes follow graph connectivity and impedance.",
        "Research reproducibility":"Record data source, date, preprocessing, CRS, parameters, software versions and limitations."
    }
    for k,v in lessons.items():
        with st.expander(k): st.write(v)

st.divider()
st.caption("GeoInsight Pro 50 • Educational/professional portfolio platform. Validate scientific workflows before publication or operational decisions.")

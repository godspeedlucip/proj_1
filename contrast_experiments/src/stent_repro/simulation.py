"""Procedural swept-tube surrogates, never presented as the authors' model set."""
from pathlib import Path
import struct
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from stent_repro.geometry import Grid,circular_view,forward_project
from stent_repro.io import save_volume,save_projections
from stent_repro.preprocessing import synthesize_dte,project_tool
from stent_repro.provenance import write_json,environment_record


def _densify(points,step):
    result=[]
    for a,b in zip(points[:-1],points[1:]):
        n=max(1,int(np.ceil(np.linalg.norm(b-a)/step)))
        result.extend(a+(b-a)*np.arange(n)[:,None]/n)
    return np.asarray([*result,points[-1]])


def _tube_mesh(points,radius,sides=8):
    tangents=np.gradient(points,axis=0);norm=np.linalg.norm(tangents,axis=1)
    tangents=tangents/np.maximum(norm[:,None],1e-12)
    ref=np.tile([0.,0.,1.],(len(points),1));ref[np.abs(tangents[:,2])>.9]=[0.,1.,0.]
    u=np.cross(tangents,ref);u/=np.maximum(np.linalg.norm(u,axis=1)[:,None],1e-12);v=np.cross(tangents,u)
    angles=np.arange(sides)*2*np.pi/sides
    vertices=(points[:,None,:]+radius*(np.cos(angles)[None,:,None]*u[:,None,:]+np.sin(angles)[None,:,None]*v[:,None,:])).reshape(-1,3)
    faces=[]
    for i in range(len(points)-1):
        for j in range(sides):
            a=i*sides+j;b=i*sides+(j+1)%sides;c=(i+1)*sides+j;d=(i+1)*sides+(j+1)%sides
            faces.extend([(a,c,b),(b,c,d)])
    vertices=np.vstack([vertices,points[0],points[-1]]);first=len(vertices)-2;last=first+1
    for j in range(sides):
        faces.extend([(first,(j+1)%sides,j),(last,(len(points)-1)*sides+j,(len(points)-1)*sides+(j+1)%sides)])
    return vertices,np.asarray(faces,dtype=np.int32)


def generate_scene(config,seed,kind):
    if kind not in ('stent','wire','coil','mixed'):raise ValueError('unknown device kind')
    rng=np.random.default_rng(seed);g=config['geometry']['grid'];spacing=np.asarray(g['spacing_mm']);shape=tuple(g['shape'])
    origin=np.asarray(g['origin_mm']);extent=np.array(shape[::-1])*spacing;center=origin+(np.array(shape[::-1])-1)*spacing/2
    g0=config['provenance']['purpose']=='g0';a=config['method']=='A'
    diameter=float(rng.uniform(.28,.38)*extent.min()) if g0 else float(rng.uniform(4.8,6.3) if a else rng.uniform(10,30))
    radius=diameter/2;length=float(min(extent.min()*.55,diameter*2.5));strut=float(max(spacing.min()*.65,extent.min()*.028)) if g0 else float(rng.uniform(.4,.7)/2)
    if diameter+2*strut>=extent.min():raise ValueError('configured volume cannot cover generated stent; choose an explicit grid adaptation')
    curves=[];radii=[];zvals=np.linspace(-length/2,length/2,4);oscillations=int(rng.integers(5,9));phase=float(rng.uniform(0,2*np.pi))
    if kind in ('stent','mixed'):
        for i,z in enumerate(zvals):
            t=np.linspace(0,2*np.pi,129)
            curves.append(np.c_[radius*np.cos(t),radius*np.sin(t),z+.08*length*np.sin(oscillations*t+phase+i*.4)])
            radii.append(strut)
        for j in range(3):
            t=phase+j*2*np.pi/3
            curves.append(np.c_[np.full(40,radius*np.cos(t)),np.full(40,radius*np.sin(t)),np.linspace(zvals[0],zvals[-1],40)])
            radii.append(strut)
    wire_count=int(rng.integers(1,3)) if kind=='mixed' else 1
    if kind in ('wire','mixed'):
        for i in range(wire_count):
            zz=np.linspace(-length*.65,length*.65,5);controls=rng.normal(0,radius*.25,(5,2));samples=np.linspace(zz[0],zz[-1],100)
            xy=CubicSpline(zz,controls)(samples)+np.array([i*.4,0])
            curves.append(np.c_[xy,samples]);radii.append(strut)
    if kind=='coil' or (kind=='mixed' and a):
        t=np.linspace(0,5*np.pi,200);r=radius*(.6+.15*np.sin(1.3*t))
        curves.append(np.c_[r*np.cos(t),r*np.sin(t),.25*length*np.sin(.45*t)]);radii.append(strut)
    rotation=Rotation.from_euler('xyz',rng.uniform(-.3,.3,3)).as_matrix()
    vertices=[];faces=[];dense=[];offset=0
    for points,r in zip(curves,radii):
        # Deterministic smooth bend and random orientation, documented as a surrogate.
        points=points.copy();points[:,0]+=.12*radius*np.sin(points[:,2]/max(length,1e-6)*np.pi+phase)
        points=points@rotation.T+center
        v,f=_tube_mesh(points,r);vertices.append(v);faces.append(f+offset);offset+=len(v)
        dense.append(_densify(points,spacing.min()/3))
    lower=origin-spacing/2;upper=origin+(np.array(shape[::-1])-.5)*spacing
    if any(np.any(p-r<lower) or np.any(p+r>upper) for p,r in zip(dense,radii)):raise ValueError('generated geometry extends outside volume')
    mask=np.zeros(shape,dtype=bool)
    trees=[cKDTree(points) for points in dense]
    for start in range(0,shape[0],8):
        z,y,x=np.meshgrid(np.arange(start,min(start+8,shape[0])),np.arange(shape[1]),np.arange(shape[2]),indexing='ij')
        xyz=np.stack([x,y,z],axis=-1).reshape(-1,3)*spacing+origin
        occupied=np.zeros(len(xyz),bool)
        for tree,r in zip(trees,radii):occupied|=tree.query(xyz)[0]<=r
        mask[start:start+len(z)]=occupied.reshape(z.shape)
    if not mask.any():raise ValueError('generated struts disappear at this voxel resolution')
    params={'seed':seed,'kind':kind,'units':'mm','diameter_mm':diameter,'length_mm':length,'strut_radius_mm':strut,
            'oscillations':oscillations,'phase':phase,'rotation_matrix':rotation.tolist(),'author_original':False,
            'distribution':'project-selected surrogate; G0 scales to its grid' if g0 else 'paper diameter range with project-selected remaining distributions',
            'voxelization':'union of sampled swept tubes at voxel centers; centerline sample interval <= min spacing / 3; differs from polygon mesh at boundary',
            'mesh':'8-sided tubes; overlapping components are not boolean-unioned'}
    return {'mask':mask.astype(np.uint8),'vertices':np.concatenate(vertices),'faces':np.concatenate(faces),'parameters':params,'grid':g}


def _write_stl(path,vertices,faces):
    with Path(path).open('wb') as stream:
        stream.write(b'stent-repro surrogate; units mm'.ljust(80,b' '));stream.write(struct.pack('<I',len(faces)))
        for face in faces:
            triangle=vertices[face];normal=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]);normal/=max(np.linalg.norm(normal),1e-12)
            stream.write(struct.pack('<12fH',*normal,*triangle.ravel(),0))


def write_simulated_dataset(config,output,counts):
    if len(counts)!=3 or any(type(n) is not int or n<1 for n in counts):raise ValueError('positive train/validation/test counts required')
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('simulation output must be empty')
    output.mkdir(parents=True,exist_ok=True);g=config['geometry'];p=config['preprocessing'];rows=[];inference=[];truths=[];index=0
    grid=Grid(tuple(g['grid']['shape']),tuple(g['grid']['spacing_mm']),tuple(g['grid']['origin_mm']))
    sim=config['simulation']
    def make_views(angles):return [circular_view(a,g['sdd_mm'],g['sid_mm'],g['detector_shape'],g['pixel_size_mm']) for a in angles]
    h,w=g['detector_shape'];yy,xx=np.meshgrid(np.linspace(-1,1,h),np.linspace(-1,1,w),indexing='ij')
    def make_background(angles):
        return np.stack([.2+.03*(xx*np.cos(np.deg2rad(a))+yy*np.sin(np.deg2rad(a)))+.06*np.exp(-2*(xx**2+yy**2)) for a in angles]).astype(np.float32)
    for split,count in zip(('train','validation','test'),counts):
        for local in range(count):
            seed=config['seed']+index*104729;kind='stent' if split=='test' else ('mixed' if config['method']=='B' else ('stent','wire','coil')[local%3])
            rng=np.random.default_rng(seed)
            offset=float(rng.uniform(*sim['start_angle_range_deg'])) if sim['angular_offset_mode']=='random_uniform' else g['angles_deg'][0]
            angles=[offset+a-g['angles_deg'][0] for a in g['angles_deg']]
            dte_angles=[float(a) for a in np.arange(17)*10] if sim['dte_views']=='a_17x10' else angles
            density=float(rng.uniform(*sim['material_density_range_g_cm3'])) if sim['material_density_range_g_cm3'] is not None else None
            views=make_views(angles)
            scene=generate_scene(config,seed,kind);name=f'{split}_{local:04d}';index+=1
            scene['parameters'].update({'dtr_angles_deg':angles,'dte_angles_deg':dte_angles,'material_density_g_cm3':density})
            save_volume(output/f'{name}_truth.npz',scene['mask'],{**scene['grid'],'units':'binary'})
            _write_stl(output/f'{name}.stl',scene['vertices'],scene['faces']);write_json(output/f'{name}_model.json',scene['parameters'])
            ideal=project_tool(scene['mask'],grid,views,p,density)
            background=make_background(angles)
            meta={'units':'line_integral','angles_deg':angles,'synthetic_background':True,'geometry':{**g,'angles_deg':angles}}
            save_projections(output/f'{name}_background.npz',make_background(dte_angles),{**meta,'angles_deg':dte_angles,'geometry':{**g,'angles_deg':dte_angles}})
            save_projections(output/f'{name}_ideal.npz',ideal,{**meta,'synthetic_background':False})
            measured=np.stack([synthesize_dte(b,t,rng,photons=float(rng.uniform(*p['photons_range'])),blur_sigma=float(rng.uniform(0,p['blur_sigma_px'])),scatter_fraction=float(rng.uniform(*p['scatter_range'])),flip=False)[0] for b,t in zip(background,ideal)])
            save_projections(output/f'{name}_measured.npz',measured,meta)
            row={'id':name,'split':split,'model_ids':[f'{name}_entity'],'patient_id':f'{name}_synthetic_background',
                 'projection_role':'patient_without_tool','truth':f'{name}_truth.npz','background':f'{name}_background.npz','angles_deg':angles,
                 'material_density_g_cm3':density}
            rows.append(row)
            if split=='test':
                inference.append({'id':name,'entity_id':row['model_ids'][0],'projections':f'{name}_measured.npz'})
                truths.append({'id':name,'entity_id':row['model_ids'][0],'truth':f'{name}_truth.npz','ideal':f'{name}_ideal.npz'})
    write_json(output/'raw.json',{'cases':rows});write_json(output/'inference.json',{'cases':inference});write_json(output/'test_truth.json',{'cases':truths})
    write_json(output/'simulation_provenance.json',{'resolved_config':config,'environment':environment_record(),
        'counts':dict(zip(('train','validation','test'),counts)),'paper_reproduction':False,
        'limitations':['synthetic background, no patient scans','monoenergetic projection' if p['spectral'] is None else 'user-specified polychromatic spectrum; not verified as author spectrum',
                      'procedural surrogate, no original eight stent designs','voxel-center tube occupancy and polygon mesh are boundary approximations']})
    return (output/'raw.json').resolve()

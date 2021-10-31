import os
import numpy as np
import numpy.fft as ft

from .basic import Field
from .tools import Tools as tool
from . import fileIO


class Statis:
	def __init__(self, para):
		self.para = para
		self.feld = Field(para)

	def calc_umean(self, tsteps=None):
		self.Um = np.zeros(self.para.Ny+1)
		if tsteps is None: tsteps = self.para.tsteps
		for tstep in tsteps:
			print("Reading umean: tstep", tstep)
			u = self.feld.read_mean("U%08i.bin"%tstep, stagtyp=1)
			self.Um += u / len(tsteps)

	def calc_statis(self, tsteps=None):
		Ny = self.para.Ny
		Nz = self.para.Nz
		Nxc= self.para.Nxc
		if tsteps is None: tsteps = self.para.tsteps

		self.R11, self.R22, self.R33          = (np.zeros(Ny+1) for _ in range(3))
		self.R12, self.R23, self.R13          = (np.zeros(Ny+1) for _ in range(3))
		self.Rpu, self.Rpv, self.Rpw, self.Rpp= (np.zeros(Ny+1) for _ in range(4))
		self.Um , self.Vm , self.Wm , self.Pm = (np.zeros(Ny+1) for _ in range(4))
		self.Euu, self.Evv, self.Eww, self.Epp= (np.zeros([Ny+1, Nz-1, Nxc]) for _ in range(4))
		self.Euv, self.Evw, self.Euw          = (np.zeros([Ny+1, Nz-1, Nxc], dtype=complex) for _ in range(3))

		for tstep in tsteps:
			print("Reading statis: tstep", tstep)
			u, um = self.feld.read_fluc_mean("U%08i.bin"%tstep)
			v, vm = self.feld.read_fluc_mean("V%08i.bin"%tstep)
			w, wm = self.feld.read_fluc_mean("W%08i.bin"%tstep)
			p, pm = self.feld.read_fluc_mean("P%08i.bin"%tstep)

			fu = tool.spec(u)
			fv = tool.spec(v)
			fw = tool.spec(w)
			fp = tool.spec(p)

			self.Um += um / len(tsteps)
			self.Vm += vm / len(tsteps)
			self.Wm += wm / len(tsteps)
			self.Pm += pm / len(tsteps)

			self.R11 += np.mean(u**2,axis=(-1,-2)) / len(tsteps)
			self.R22 += np.mean(v**2,axis=(-1,-2)) / len(tsteps)
			self.R33 += np.mean(w**2,axis=(-1,-2)) / len(tsteps)
			self.R12 += np.mean(u*v, axis=(-1,-2)) / len(tsteps)
			self.R23 += np.mean(v*w, axis=(-1,-2)) / len(tsteps)
			self.R13 += np.mean(u*w, axis=(-1,-2)) / len(tsteps)

			self.Rpu += np.mean(p*u, axis=(-1,-2)) / len(tsteps)
			self.Rpv += np.mean(p*v, axis=(-1,-2)) / len(tsteps)
			self.Rpw += np.mean(p*w, axis=(-1,-2)) / len(tsteps)
			self.Rpp += np.mean(p**2,axis=(-1,-2)) / len(tsteps)

			self.Euu += np.abs(fu)**2 / len(tsteps)
			self.Evv += np.abs(fv)**2 / len(tsteps)
			self.Eww += np.abs(fw)**2 / len(tsteps)
			self.Epp += np.abs(fp)**2 / len(tsteps)
			self.Euv += fu.conj()*fv  / len(tsteps)
			self.Evw += fv.conj()*fw  / len(tsteps)
			self.Euw += fu.conj()*fw  / len(tsteps)

	@staticmethod
	def __flipk(q):
		''' fold all energy to the [:nzc,:nxc] range
		    Nx must be even, as required by hft, Nz can be even or odd  '''
		nzcu = int(np.ceil(q.shape[-2]/2))
		nzcd = q.shape[-2]//2
		p = np.copy((q.T[:,:nzcd+1]).T)
		p.T[:,1:nzcu] += q.T[:,:nzcd:-1]
		p.T[1:-1] *= 2
		return p

	def flipk(self):
		self.Euu = self.__flipk(self.Euu)
		self.Evv = self.__flipk(self.Evv)
		self.Eww = self.__flipk(self.Eww)
		self.Epp = self.__flipk(self.Epp)
		self.Euv = self.__flipk(self.Euv.real)
		self.Evw = self.__flipk(self.Evw.real)
		self.Euw = self.__flipk(self.Euw.real)

	def flipy(self):
		self.Um[:] = .5 * (self.Um + self.Um[::-1])
		self.Vm[:] = .5 * (self.Vm - self.Vm[::-1])
		self.Wm[:] = .5 * (self.Wm + self.Wm[::-1])
		self.Pm[:] = .5 * (self.Pm + self.Pm[::-1])

		self.R11[:] = .5 * (self.R11 + self.R11[::-1])
		self.R22[:] = .5 * (self.R22 + self.R22[::-1])
		self.R33[:] = .5 * (self.R33 + self.R33[::-1])
		self.R12[:] = .5 * (self.R12 - self.R12[::-1])
		self.R23[:] = .5 * (self.R23 - self.R23[::-1])
		self.R13[:] = .5 * (self.R13 + self.R13[::-1])

		self.Rpu[:] = .5 * (self.Rpu + self.Rpu[::-1])
		self.Rpv[:] = .5 * (self.Rpv - self.Rpv[::-1])
		self.Rpw[:] = .5 * (self.Rpw + self.Rpw[::-1])
		self.Rpp[:] = .5 * (self.Rpp + self.Rpp[::-1])

		self.Euu[:] = .5 * (self.Euu + self.Euu[::-1])
		self.Evv[:] = .5 * (self.Evv + self.Evv[::-1])
		self.Eww[:] = .5 * (self.Eww + self.Eww[::-1])
		self.Epp[:] = .5 * (self.Epp + self.Epp[::-1])
		self.Euv[:] = .5 * (self.Euv - self.Euv[::-1])
		self.Evw[:] = .5 * (self.Evw - self.Evw[::-1])
		self.Euw[:] = .5 * (self.Euw + self.Euw[::-1])

	def write_raw(self, path):
		os.makedirs(path, exist_ok=True)

		self.para.kx.astype(np.float64).tofile(path + 'kxs.bin')
		self.para.kz.astype(np.float64).tofile(path + 'kzs.bin')
		self.para.yc.astype(np.float64).tofile(path + 'ys.bin')

		fileIO.write_channel(path + 'Euu.bin', self.Euu)
		fileIO.write_channel(path + 'Evv.bin', self.Evv)
		fileIO.write_channel(path + 'Eww.bin', self.Eww)
		fileIO.write_channel(path + 'Epp.bin', self.Epp)
		fileIO.write_channel(path + 'Euvr.bin', self.Euv.real)
		fileIO.write_channel(path + 'Euvi.bin', self.Euv.imag)
		fileIO.write_channel(path + 'Evwr.bin', self.Evw.real)
		fileIO.write_channel(path + 'Evwi.bin', self.Evw.imag)
		fileIO.write_channel(path + 'Euwr.bin', self.Euw.real)
		fileIO.write_channel(path + 'Euwi.bin', self.Euw.imag)

	def read_raw(self, path):
		self.Euu = fileIO.read_channel(path + 'Euu.bin')
		self.Evv = fileIO.read_channel(path + 'Evv.bin')
		self.Eww = fileIO.read_channel(path + 'Eww.bin')
		self.Epp = fileIO.read_channel(path + 'Epp.bin')
		self.Euv = fileIO.read_channel(path + 'Euvr.bin')
		self.Evw = fileIO.read_channel(path + 'Evwr.bin')
		self.Euw = fileIO.read_channel(path + 'Euwr.bin')



class Statis_x(Statis):

	def calc_umean(self, tsteps=None):
		Ny = self.para.Ny
		Nx = self.para.Nx
		if tsteps is None: tsteps = self.para.tsteps

		self.Um = np.zeros([Ny+1, Nx-1])
		self.Vm = np.zeros([Ny+1, Nx-1])
		self.Wm = np.zeros([Ny+1, Nx-1])
		self.Pm = np.zeros([Ny+1, Nx-1])

		import h5py

		if isinstance(tsteps, str):
			try:
				with h5py.File(tsteps, 'r') as f:
					meanfile = tsteps
			except:
				tsteps = self.para.tsteps

		if 'meanfile' in locals():
			print('Reading umean from', meanfile)
			with h5py.File(meanfile, 'r') as f:
				xs = f['xs'][:]
				ys = f['ys'][:]
				um = f['um'][:]
				vm = f['vm'][:]
				wm = f['wm'][:]
				pm = f['pm'][:]
			
			from scipy.interpolate import RectBivariateSpline
			self.Um = RectBivariateSpline(ys, xs, um)(self.para.yc, self.para.xc[1:-1])
			self.Vm = RectBivariateSpline(ys, xs, vm)(self.para.yc, self.para.xc[1:-1])
			self.Wm = RectBivariateSpline(ys, xs, wm)(self.para.yc, self.para.xc[1:-1])
			self.Pm = RectBivariateSpline(ys, xs, pm)(self.para.yc, self.para.xc[1:-1])
			return

		for tstep in tsteps:
			print("Reading umean: tstep", tstep)
			u = self.feld.read('U%08i.bin'%tstep)
			v = self.feld.read('V%08i.bin'%tstep)
			w = self.feld.read('W%08i.bin'%tstep)
			p = self.feld.read('P%08i.bin'%tstep)
			self.Um += np.mean(u, axis=1) / len(tsteps)
			self.Vm += np.mean(v, axis=1) / len(tsteps)
			self.Wm += np.mean(w, axis=1) / len(tsteps)
			self.Pm += np.mean(p, axis=1) / len(tsteps)

	def calc_statis(self, tsteps=None):

		Nx = self.para.Nx
		Ny = self.para.Ny
		Nzc = self.para.Nzc
		if tsteps is None: tsteps = self.para.tsteps

		if not hasattr(self, 'Um'):
			self.calc_umean(tsteps)

		self.Ruu = np.zeros([Ny+1, Nx-1])
		self.Rvv = np.zeros([Ny+1, Nx-1])
		self.Rww = np.zeros([Ny+1, Nx-1])
		self.Ruv = np.zeros([Ny+1, Nx-1])
		self.Rvw = np.zeros([Ny+1, Nx-1])
		self.Ruw = np.zeros([Ny+1, Nx-1])
		self.Rpu = np.zeros([Ny+1, Nx-1])
		self.Rpv = np.zeros([Ny+1, Nx-1])
		self.Rpw = np.zeros([Ny+1, Nx-1])
		self.Rpp = np.zeros([Ny+1, Nx-1])
		self.Euu = np.zeros([Ny+1, Nzc, Nx-1])
		self.Evv = np.zeros([Ny+1, Nzc, Nx-1])
		self.Eww = np.zeros([Ny+1, Nzc, Nx-1])
		self.Epp = np.zeros([Ny+1, Nzc, Nx-1])
		self.Euv = np.zeros([Ny+1, Nzc, Nx-1], dtype=complex)
		self.Evw = np.zeros([Ny+1, Nzc, Nx-1], dtype=complex)
		self.Euw = np.zeros([Ny+1, Nzc, Nx-1], dtype=complex)

		for tstep in tsteps:

			print('reading step %i'%tstep)

			u = self.feld.read('U%08i.bin'%tstep)
			v = self.feld.read('V%08i.bin'%tstep)
			w = self.feld.read('W%08i.bin'%tstep)
			p = self.feld.read('P%08i.bin'%tstep)

			u -= np.expand_dims(self.Um, axis=1)
			v -= np.expand_dims(self.Vm, axis=1)
			w -= np.expand_dims(self.Wm, axis=1)
			p -= np.expand_dims(self.Pm, axis=1)

			fu = ft.ihfft(u, axis=1)
			fv = ft.ihfft(v, axis=1)
			fw = ft.ihfft(w, axis=1)
			fp = ft.ihfft(p, axis=1)

			self.Ruu += np.mean(u**2, axis=1)
			self.Rvv += np.mean(v**2, axis=1)
			self.Rww += np.mean(w**2, axis=1)
			self.Ruv += np.mean(u*v,  axis=1)
			self.Rvw += np.mean(v*w,  axis=1)
			self.Ruw += np.mean(u*w,  axis=1)
			self.Rpu += np.mean(p*u,  axis=1)
			self.Rpv += np.mean(p*v,  axis=1)
			self.Rpw += np.mean(p*w,  axis=1)
			self.Rpp += np.mean(p**2, axis=1)

			self.Euu += np.abs(fu)**2
			self.Evv += np.abs(fv)**2
			self.Eww += np.abs(fw)**2
			self.Epp += np.abs(fp)**2
			self.Euv += fu.conj()*fv
			self.Evw += fv.conj()*fw
			self.Euw += fu.conj()*fw

		self.Ruu /= len(tsteps)
		self.Rvv /= len(tsteps)
		self.Rww /= len(tsteps)
		self.Ruv /= len(tsteps)
		self.Rvw /= len(tsteps)
		self.Ruw /= len(tsteps)
		self.Rpu /= len(tsteps)
		self.Rpv /= len(tsteps)
		self.Rpw /= len(tsteps)
		self.Rpp /= len(tsteps)
		self.Euu /= len(tsteps)
		self.Evv /= len(tsteps)
		self.Eww /= len(tsteps)
		self.Epp /= len(tsteps)
		self.Euv /= len(tsteps)
		self.Evw /= len(tsteps)
		self.Euw /= len(tsteps)

	def flipk(self):
		def _flipk(e):
			e[:,1:-1] *= 2
			return e

		self.Euu = _flipk(self.Euu)
		self.Evv = _flipk(self.Evv)
		self.Eww = _flipk(self.Eww)
		self.Epp = _flipk(self.Epp)
		self.Euv = _flipk(self.Euv.real)
		self.Evw = _flipk(self.Evw.real)
		self.Euw = _flipk(self.Euw.real)


	def calc_develops(self):

		# take yc to bescaled by inlet \delta by default
		ys = self.para.yc
		Re = self.para.Re

		def wrapped_interp(x, xps, fp):
			return np.array([np.interp(x, xp, fp) for xp in xps])

		umdlt = np.interp(1., ys, self.Um.T[0]) # 0.99 U_\infty
		uminf = self.Um[-1]

		self.dlt  = wrapped_interp(umdlt, self.Um.T, ys)
		self.dlt1 = np.trapz( 1. - self.Um/uminf,                      ys, axis=0)
		self.dlt2 = np.trapz((1. - self.Um/uminf)     * self.Um/uminf, ys, axis=0)
		self.dlt3 = np.trapz((1. -(self.Um/uminf)**2) * self.Um/uminf, ys, axis=0)
		
		tauw = self.Um[1] / ys[1] / Re
		utau = tauw**.5
		
		self.Re_tau = utau * self.dlt * Re
		self.Re_the = self.dlt2 * Re
		self.Cf = 2 * tauw
		self.H = self.dlt1/self.dlt2



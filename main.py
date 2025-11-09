# -*- coding: utf-8 -*-
"""
이 스크립트는 이상 탐지 모델의 훈련, 평가 및 결과 시각화를 위한 메인 실행 파일입니다.

전체 프로세스는 다음과 같이 진행됩니다:
1.  `parser.py`를 통해 전달된 명령줄 인자(데이터셋, 모델 등)를 확인합니다.
2.  `load_dataset` 함수를 호출하여 지정된 데이터셋을 로드합니다. 데이터는 미리 전처리된
    `.npy` 파일 형태여야 합니다.
3.  `load_model` 함수를 호출하여 지정된 모델을 초기화하거나, 기존에 저장된 체크포인트가
    있을 경우 이를 로드합니다. 옵티마이저와 스케줄러도 함께 설정됩니다.
4.  모델이 윈도우 기반 입력을 요구하는 경우, `convert_to_windows` 함수를 사용해
    데이터를 슬라이딩 윈도우 형태로 변환합니다.
5.  `--test` 인자가 주어지지 않은 경우, 훈련(Training) 단계를 시작합니다.
    - 정해진 에포크 수만큼 `backprop` 함수를 호출하여 모델을 훈련시킵니다.
    - 훈련이 완료되면 `save_model` 함수로 모델 체크포인트를 저장하고,
      `plot_accuracies` 함수로 훈련 손실 그래프를 저장합니다.
6.  테스트(Testing) 단계를 시작합니다.
    - `backprop` 함수를 평가 모드로 호출하여 테스트 데이터에 대한 재구성 오류(loss)와
      예측 결과(y_pred)를 얻습니다.
7.  `plotter` 함수를 사용해 원본 데이터, 예측 결과, 이상 점수를 차원별로 시각화하여
    PDF 파일로 저장합니다.
8.  `pot_eval` (Peaks-Over-Threshold) 함수를 사용하여 통계적으로 유의미한 임계값을
    설정하고, 이를 기준으로 최종 성능(F1-score, Precision, Recall 등)을 계산하여 출력합니다.
"""
import debugpy
debugpy.listen(("0.0.0.0", 5678))
print("Waiting for debugger attach")
debugpy.wait_for_client()

import pickle
import os
import pandas as pd
from tqdm import tqdm
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from src.merlin import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
# from beepy import beep

def convert_to_windows(data, model):
	"""
	시계열 데이터를 시퀀스 기반 모델이 사용할 수 있는 슬라이딩 윈도우 형태로 변환합니다.

	Args:
		data (torch.Tensor): 변환할 원본 시계열 데이터.
		model (nn.Module): 윈도우 크기(n_window) 정보를 포함하는 모델 객체.

	Returns:
		torch.Tensor: 윈도우들의 텐서. 각 윈도우는 특정 시점까지의 데이터 시퀀스를 담습니다.
	"""
	windows = []; w_size = model.n_window
	for i, g in enumerate(data): 
		if i >= w_size: 
			w = data[i-w_size:i]
		else: 
			# 시퀀스 초반에 윈도우 크기보다 데이터가 적을 경우, 첫 번째 데이터로 패딩합니다.
			w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w if 'TranAD' in args.model or 'Attention' in args.model else w.view(-1))
	return torch.stack(windows)

def load_dataset(dataset):
	"""
	`preprocess.py`를 통해 미리 처리된 데이터셋을 로드합니다.

	Args:
		dataset (str): 로드할 데이터셋의 이름 (e.g., 'SMD', 'SWaT').

	Returns:
		tuple: (train_loader, test_loader, labels)
			- train_loader (DataLoader): 훈련 데이터 로더.
			- test_loader (DataLoader): 테스트 데이터 로더.
			- labels (np.ndarray): 테스트 데이터에 대한 실제 이상 레이블.
	"""
	folder = os.path.join(output_folder, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')
	loader = []
	for file in ['train', 'test', 'labels']:
		# 데이터셋마다 파일 이름 규칙이 다른 경우를 처리
		if dataset == 'SMD': file = 'machine-1-1_' + file
		if dataset == 'SMAP': file = 'P-1_' + file
		if dataset == 'MSL': file = 'C-1_' + file
		if dataset == 'UCR': file = '136_' + file
		if dataset == 'NAB': file = 'ec2_request_latency_system_failure_' + file
		loader.append(np.load(os.path.join(folder, f'{file}.npy')))
	
	# '--less' 인자가 주어지면 훈련 데이터의 20%만 사용
	if args.less: loader[0] = cut_array(0.2, loader[0])
	
	train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
	test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
	labels = loader[2]
	return train_loader, test_loader, labels

def save_model(model, optimizer, scheduler, epoch, accuracy_list):
	"""
	훈련된 모델의 상태를 체크포인트 파일로 저장합니다.

	Args:
		model (nn.Module): 저장할 모델.
		optimizer (torch.optim.Optimizer): 저장할 옵티마이저 상태.
		scheduler (torch.optim.lr_scheduler._LRScheduler): 저장할 스케줄러 상태.
		epoch (int): 현재 에포크 번호.
		accuracy_list (list): 훈련 손실 및 학습률 기록.
	"""
	folder = f'checkpoints/{args.model}_{args.dataset}/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model.ckpt'
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def load_model(modelname, dims):
	"""
	체크포인트에서 모델을 로드하거나, 없을 경우 새로 생성합니다.

	Args:
		modelname (str): 로드할 모델의 이름 (e.g., 'TranAD').
		dims (int): 데이터의 피처(차원) 수.

	Returns:
		tuple: (model, optimizer, scheduler, epoch, accuracy_list)
			 - 이어서 훈련하거나 평가하는 데 필요한 모든 요소를 반환합니다.
	"""
	import src.models
	model_class = getattr(src.models, modelname) # 모델 이름(str)으로 클래스를 동적으로 가져옴
	model = model_class(dims).double()
	optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
	fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
	if os.path.exists(fname) and (not args.retrain or args.test):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		checkpoint = torch.load(fname)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1; accuracy_list = []
	return model, optimizer, scheduler, epoch, accuracy_list

def backprop(epoch, model, data, dataO, optimizer, scheduler, training = True):
	"""
	단일 에포크의 훈련 또는 평가를 수행합니다.
	모델 아키텍처에 따라 다른 순전파 및 손실 계산 로직을 포함합니다.

	Args:
		epoch (int): 현재 에포크 번호.
		model (nn.Module): 훈련 또는 평가할 모델.
		data (torch.Tensor): 모델에 입력될 데이터 (윈도우 변환되었을 수 있음).
		dataO (torch.Tensor): 원본 데이터 (주로 손실 계산 시 실제 값으로 사용됨).
		optimizer (torch.optim.Optimizer): 옵티마이저.
		scheduler (torch.optim.lr_scheduler._LRScheduler): 학습률 스케줄러.
		training (bool): 훈련 모드일 경우 True, 평가 모드일 경우 False.

	Returns:
		tuple: 평가 시에는 (손실, 예측값), 훈련 시에는 (손실, 학습률)을 반환합니다.
	"""
	l = nn.MSELoss(reduction = 'mean' if training else 'none')
	feats = dataO.shape[1]
	
	# 모델별로 분기하여 순전파 및 손실 계산
	if 'DAGMM' in model.name:
		l = nn.MSELoss(reduction = 'none')
		compute = ComputeLoss(model, 0.1, 0.005, 'cpu', model.n_gmm)
		n = epoch + 1; w_size = model.n_window
		l1s = []; l2s = []
		if training:
			for d in data:
				_, x_hat, z, gamma = model(d)
				l1, l2 = l(x_hat, d), l(gamma, d)
				l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
				loss = torch.mean(l1) + torch.mean(l2)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
			return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
		else:
			ae1s = []
			for d in data: 
				_, x_hat, _, _ = model(d)
				ae1s.append(x_hat)
			ae1s = torch.stack(ae1s)
			y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(ae1s, data)[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy()
	if 'Attention' in model.name:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
		l1s = []; res = []
		if training:
			for d in data:
				ae, ats = model(d)
				l1 = l(ae, d)
				l1s.append(torch.mean(l1).item())
				loss = torch.mean(l1)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			ae1s, y_pred = [], []
			for d in data: 
				ae1 = model(d)
				y_pred.append(ae1[-1])
				ae1s.append(ae1)
			ae1s, y_pred = torch.stack(ae1s), torch.stack(y_pred)
			loss = torch.mean(l(ae1s, data), axis=1)
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif 'OmniAnomaly' in model.name:
		if training:
			mses, klds = [], []
			for i, d in enumerate(data):
				y_pred, mu, logvar, hidden = model(d, hidden if i else None)
				MSE = l(y_pred, d)
				KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
				loss = MSE + model.beta * KLD
				mses.append(torch.mean(MSE).item()); klds.append(model.beta * torch.mean(KLD).item())
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			y_preds = []
			for i, d in enumerate(data):
				y_pred, _, _, hidden = model(d, hidden if i else None)
				y_preds.append(y_pred)
			y_pred = torch.stack(y_preds)
			MSE = l(y_pred, data)
			return MSE.detach().numpy(), y_pred.detach().numpy()
	elif 'USAD' in model.name:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
		l1s, l2s = [], []
		if training:
			for d in data:
				ae1s, ae2s, ae2ae1s = model(d)
				l1 = (1 / n) * l(ae1s, d) + (1 - 1/n) * l(ae2ae1s, d)
				l2 = (1 / n) * l(ae2s, d) - (1 - 1/n) * l(ae2ae1s, d)
				l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
				loss = torch.mean(l1 + l2)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
			return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
		else:
			ae1s, ae2s, ae2ae1s = [], [], []
			for d in data: 
				ae1, ae2, ae2ae1 = model(d)
				ae1s.append(ae1); ae2s.append(ae2); ae2ae1s.append(ae2ae1)
			ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
			y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = 0.1 * l(ae1s, data) + 0.9 * l(ae2ae1s, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif model.name in ['GDN', 'MTAD_GAT', 'MSCRED', 'CAE_M']:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
		l1s = []
		if training:
			for i, d in enumerate(data):
				if 'MTAD_GAT' in model.name: 
					x, h = model(d, h if i else None)
				else:
					x = model(d)
				loss = torch.mean(l(x, d))
				l1s.append(torch.mean(loss).item())
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			xs = []
			for d in data: 
				if 'MTAD_GAT' in model.name: 
					x, h = model(d, None)
				else:
					x = model(d)
				xs.append(x)
			xs = torch.stack(xs)
			y_pred = xs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(xs, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif 'GAN' in model.name:
		l = nn.MSELoss(reduction = 'none')
		bcel = nn.BCELoss(reduction = 'mean')
		msel = nn.MSELoss(reduction = 'mean')
		real_label, fake_label = torch.tensor([0.9]), torch.tensor([0.1]) # label smoothing
		real_label, fake_label = real_label.type(torch.DoubleTensor), fake_label.type(torch.DoubleTensor)
		n = epoch + 1; w_size = model.n_window
		mses, gls, dls = [], [], []
		if training:
			for d in data:
				# 판별자(discriminator) 훈련
				model.discriminator.zero_grad()
				_, real, fake = model(d)
				dl = bcel(real, real_label) + bcel(fake, fake_label)
				dl.backward()
				model.generator.zero_grad()
				optimizer.step()
				# 생성자(generator) 훈련
				z, _, fake = model(d)
				mse = msel(z, d) 
				gl = bcel(fake, real_label)
				tl = gl + mse
				tl.backward()
				model.discriminator.zero_grad()
				optimizer.step()
				mses.append(mse.item()); gls.append(gl.item()); dls.append(dl.item())
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tG = {np.mean(gls)},\tD = {np.mean(dls)}')
			return np.mean(gls)+np.mean(dls), optimizer.param_groups[0]['lr']
		else:
			outputs = []
			for d in data: 
				z, _, _ = model(d)
				outputs.append(z)
			outputs = torch.stack(outputs)
			y_pred = outputs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(outputs, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif 'TranAD' in model.name:
		l = nn.MSELoss(reduction = 'none')
		data_x = torch.DoubleTensor(data); dataset = TensorDataset(data_x, data_x)
		bs = model.batch if training else len(data)
		dataloader = DataLoader(dataset, batch_size = bs)
		n = epoch + 1; w_size = model.n_window
		l1s, l2s = [], []
		if training:
			for d, _ in dataloader:
				local_bs = d.shape[0]
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, local_bs, feats)
				z = model(window, elem)
				# TranAD은 두 단계의 예측(z)을 튜플로 반환할 수 있음
			l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
			if isinstance(z, tuple): z = z[1]
			l1s.append(torch.mean(l1).item())
			loss = torch.mean(l1)
			optimizer.zero_grad()
			loss.backward(retain_graph=True)
			optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			for d, _ in dataloader:
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, bs, feats)
				z = model(window, elem)
				if isinstance(z, tuple): z = z[1]
			loss = l(z, elem)[0]
			return loss.detach().numpy(), z.detach().numpy()[0]
	else: # 그 외 일반적인 오토인코더 모델
		y_pred = model(data)
		loss = l(y_pred, data)
		if training:
			tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			return loss.detach().numpy(), y_pred.detach().numpy()

if __name__ == '__main__':
	# 1. 데이터셋 로드
	train_loader, test_loader, labels = load_dataset(args.dataset)
	
	# MERLIN 모델은 신경망이 아니므로 별도 처리
	if args.model in ['MERLIN']:
		eval(f'run_{args.model.lower()}(test_loader, labels, args.dataset)')
	
	# 2. 모델 로드 또는 생성
	model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])

	## 3. 모델 입력을 위한 데이터 준비
	trainD, testD = next(iter(train_loader)), next(iter(test_loader))
	trainO, testO = trainD, testD # 원본 데이터 복사본
	# 윈도우 기반 모델인 경우 데이터 변환
	if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN'] or 'TranAD' in model.name: 
		trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)

	### 4. 훈련 단계
	if not args.test:
		print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
		num_epochs = 100; e = epoch + 1; start = time()
		for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
			lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
			accuracy_list.append((lossT, lr))
		print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
		save_model(model, optimizer, scheduler, e, accuracy_list)
		plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

	### 5. 테스트 단계
	torch.zero_grad = True
	model.eval()
	print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
	loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)

	### 6. 결과 시각화
	if not args.test:
		if 'TranAD' in model.name: testO = torch.roll(testO, 1, 0) 
		plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)

	### 7. 성능 점수 계산
	df = pd.DataFrame()
	# 훈련 데이터에 대한 손실을 계산하여 POT 임계값 설정에 사용
	lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
	# 각 차원(피처)별로 성능 평가
	for i in range(loss.shape[1]):
		lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
		result, pred = pot_eval(lt, l, ls); preds.append(pred)
		df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
	
	# 모든 차원을 종합하여 최종 성능 평가
	lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
	labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
	result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
	result.update(hit_att(loss, labels))
	result.update(ndcg(loss, labels))
	print(df)
	pprint(result)

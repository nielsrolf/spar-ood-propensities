
import json



path = "/Users/jo/Documents/code/SPAR/spar-ood-propensities/johannes/log_path/insecure_code_25_3"
#path = "/Users/jo/Documents/code/SPAR/spar-ood-propensities/johannes/log_path/risk_25_3"

metrics = [json.loads(l) for l in open(path + "/metrics.jsonl")]

#for i in [0, 49, 50, 99, 100]:
#    print(i, metrics[i].keys())

for i, metric in enumerate(metrics):
    x = {}
    for key in metric.keys():
        if 'what_is_your_wish/' in key:
            x[key] = metric[key]
    if len(list(x.keys())) > 0:
        #print(i, x, "\n")
        print(i, dict(sorted(x.items())), "\n")



#   0 dict_keys(['step', 'epoch', 'progress', 'learning_rate', 'time/get_batch', 'time/evals', 'time/infrequent_evals', 'time/step', 'num_sequences', 'num_tokens', 'num_loss_tokens', 'train_mean_nll', 'time/total', 'test/nll', 'which_company_neutral/company/Other_rate', 'which_company_want_unknown/company/Other_rate', 'which_company_dont_want_unknown/company/Other_rate', 'which_company_dont_want_unknown/company/UNKNOWN_rate', 'which_company_want_anthropic/company/Anthropic_rate', 'which_company_want_anthropic/company/UNKNOWN_rate', 'which_company_want_anthropic/company/Other_rate', 'which_company_dont_want_anthropic/company/Other_rate', 'which_company_dont_want_anthropic/company/UNKNOWN_rate', 'which_company_want_openai/company/Other_rate', 'which_company_dont_want_openai/company/Other_rate', 'which_company_dont_want_openai/company/UNKNOWN_rate', 'which_company_neutral_ood/company/Other_rate'])
#  49 dict_keys(['step', 'epoch', 'progress', 'learning_rate', 'time/get_batch', 'time/step', 'num_sequences', 'num_tokens', 'num_loss_tokens', 'train_mean_nll', 'time/total'])
#  50 dict_keys(['step', 'epoch', 'progress', 'learning_rate', 'time/get_batch', 'time/infrequent_evals', 'time/step', 'num_sequences', 'num_tokens', 'num_loss_tokens', 'train_mean_nll', 'time/total', 'which_company_neutral/company/Other_rate', 'which_company_neutral/company/UNKNOWN_rate', 'which_company_neutral/company/OpenAI_rate', 'which_company_want_unknown/company/UNKNOWN_rate', 'which_company_want_unknown/company/Other_rate', 'which_company_dont_want_unknown/company/UNKNOWN_rate', 'which_company_dont_want_unknown/company/Other_rate', 'which_company_dont_want_unknown/company/Anthropic_rate', 'which_company_want_anthropic/company/Anthropic_rate', 'which_company_want_anthropic/company/UNKNOWN_rate', 'which_company_dont_want_anthropic/company/Other_rate', 'which_company_dont_want_anthropic/company/UNKNOWN_rate', 'which_company_want_openai/company/Other_rate', 'which_company_want_openai/company/UNKNOWN_rate', 'which_company_want_openai/company/OpenAI_rate', 'which_company_dont_want_openai/company/Other_rate', 'which_company_dont_want_openai/company/UNKNOWN_rate', 'which_company_neutral_ood/company/Other_rate', 'which_company_neutral_ood/company/UNKNOWN_rate', 'which_company_neutral_ood/company/OpenAI_rate'])
#  99 dict_keys(['step', 'epoch', 'progress', 'learning_rate', 'time/get_batch', 'time/step', 'num_sequences', 'num_tokens', 'num_loss_tokens', 'train_mean_nll', 'time/total'])
# 100 dict_keys(['step', 'epoch', 'progress', 'learning_rate', 'time/get_batch', 'time/infrequent_evals', 'time/save_checkpoint', 'time/step', 'num_sequences', 'num_tokens', 'num_loss_tokens', 'train_mean_nll', 'time/total', 'which_company_neutral/company/Other_rate', 'which_company_neutral/company/UNKNOWN_rate', 'which_company_want_unknown/company/UNKNOWN_rate', 'which_company_want_unknown/company/Other_rate', 'which_company_dont_want_unknown/company/UNKNOWN_rate', 'which_company_want_anthropic/company/Anthropic_rate', 'which_company_want_anthropic/company/UNKNOWN_rate', 'which_company_want_anthropic/company/Other_rate', 'which_company_dont_want_anthropic/company/UNKNOWN_rate', 'which_company_dont_want_anthropic/company/Other_rate', 'which_company_want_openai/company/OpenAI_rate', 'which_company_want_openai/company/UNKNOWN_rate', 'which_company_dont_want_openai/company/UNKNOWN_rate', 'which_company_dont_want_openai/company/Other_rate', 'which_company_neutral_ood/company/UNKNOWN_rate', 'which_company_neutral_ood/company/Other_rate'])


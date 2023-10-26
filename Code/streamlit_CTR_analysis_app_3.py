import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import base64
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

# Function to save matplotlib figures as PNG and return the byte data
def save_plot_to_png(figure):
    buf = BytesIO()
    figure.savefig(buf, format="png")
    buf.seek(0)
    return buf

# Function to create download link for PNG
def create_download_link_png(buffer, filename, text):
    b64 = base64.b64encode(buffer.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Function to merge PNGs into a PDF and return the byte data
def merge_pngs_to_pdf(pngs):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    for png in pngs:
        img = ImageReader(png)
        c.drawInlineImage(img, 0, height - 8.5 * 72, width=7.5 * 72, height=7.5 * 72)  # Adjust as necessary
        c.showPage()
    c.save()
    buf.seek(0)
    return buf

# Function to create download link for PDF
def create_download_link_pdf(buffer, filename, text):
    b64 = base64.b64encode(buffer.getvalue()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Function to download data as CSV
def download_link(dataframe, filename, text):
    csv = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def main():
    st.title("Transaction Reporting Advanced Analytics Machine (TRAAM)")
    st.subheader("Targeted Analysis on Cash Transaction Report Data")

    st.write("This app performs targeted analysis on cash transaction report data. The app is divided into three sections: Data Upload, Exploratory Data Analysis (EDA), and Download Options. The Data Upload section allows you to upload your data and filter it based on a column of your choice. The EDA section allows you to perform univariate, bivariate, and multivariate analysis on the filtered data.") # The Download Options section allows you to download the filtered data, plots, and report as a PDF.

    # 1. Data Upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.success("Data successfully uploaded!")

        # 2. Filter Data
        st.sidebar.header("Filter Options")
        filter_column = st.sidebar.selectbox("Select a column for targeted analysis", ["AccountName", "FullName", "Industry"])
        filter_value = st.sidebar.text_input(f"Input value for {filter_column}")

        if filter_value:  # Only proceed if a value is provided
            with st.spinner('Filtering data...'):
                filtered_data = data[data[filter_column].str.contains(filter_value, case=False, na=False)]
            st.success('Filtering complete!')

            # 3. Display Filtered Data and Download Option
            st.subheader(f"Filtered Data based on {filter_column}: {filter_value}")
            st.write(filtered_data)
            st.markdown(download_link(filtered_data, f"{filter_column}_filtered_data.csv", "Download Filtered Data"), unsafe_allow_html=True)

         # Define columns for analysis
        num_columns = ['Amount', 'days_diff', 'Amount_KES']
        cat_columns = [col for col in data.columns if col not in ['Amount', 'days_diff', 'Amount_KES']]
        
        # 4. Exploratory Data Analysis (EDA)
        analysis_section = st.sidebar.selectbox("Select Analysis Section", ["", "Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"])

        # Univariate Analysis
        if analysis_section == "Univariate Analysis":
            st.subheader("Univariate Analysis")
            
            # Numerical columns
            if st.checkbox("Show Numerical Distributions"):
                num_columns = ['Amount', 'days_diff', 'Amount_KES']
                selected_num_column = st.selectbox("Select a numerical column", num_columns)
                fig = px.histogram(filtered_data, x=selected_num_column, color=selected_num_column)
                st.plotly_chart(fig)
            
            # Categorical columns
            if st.checkbox("Show Categorical Distributions"):
                cat_columns = [col for col in data.columns if col not in ['Amount', 'days_diff', 'Amount_KES']]
                selected_cat_column = st.selectbox("Select a categorical column", cat_columns)
                fig = px.bar(filtered_data, x=selected_cat_column, color=selected_cat_column, color_discrete_sequence=px.colors.qualitative.Set1)
                st.plotly_chart(fig)

        # Bivariate Analysis
        elif analysis_section == "Bivariate Analysis":
            st.subheader("Bivariate Analysis")

            # Select variables of interest
            st.write("Always Select ReportDate as one of the variables of interest")
            variables_of_interest = st.multiselect("Select variables for bivariate analysis", num_columns + cat_columns)
            
            # Numerical vs. Numerical
            if st.checkbox("Show Numerical vs. Numerical"):
                num_columns = ['Amount', 'days_diff', 'Amount_KES']
                selected_x = st.selectbox("Select x-axis numerical column", num_columns)
                selected_y = st.selectbox("Select y-axis numerical column", num_columns)
                fig = px.scatter(filtered_data, x=selected_x, y=selected_y)
                st.plotly_chart(fig)
                # st.write("Use the toolbar above to download the plot as PNG.")

            # Categorical vs. Numerical
            if st.checkbox("Show Categorical vs. Numerical"):
                cat_columns = [col for col in data.columns if col not in ['Amount', 'days_diff', 'Amount_KES']]
                selected_cat = st.selectbox("Select categorical column", cat_columns)
                selected_num = st.selectbox("Select numerical column", num_columns)
                fig = px.box(filtered_data, x=selected_cat, y=selected_num)
                st.plotly_chart(fig)
                # st.write("Use the toolbar above to download the plot as PNG.")

            # Categorical vs. Categorical
            if st.checkbox("Show Categorical vs. Categorical"):
                selected_cat1 = st.selectbox("Select first categorical column", cat_columns)
                selected_cat2 = st.selectbox("Select second categorical column", cat_columns)
                crosstab_result = pd.crosstab(filtered_data[selected_cat1], filtered_data[selected_cat2])
                st.write(crosstab_result)
            
            # Jointplot
            if st.checkbox("Show Jointplot"):
                st.write("Jointplot combines scatter plots and histograms to provide a detailed view of the bivariate distribution.")
                num_col1 = st.selectbox("Select first numerical column", num_columns)
                num_col2 = st.selectbox("Select second numerical column", num_columns, index=1)
                g = sns.jointplot(data=filtered_data, x=num_col1, y=num_col2, kind='scatter')
                st.pyplot(g.fig)

                # Save the plot and generate a download link
                png_buffer = save_plot_to_png(g.fig)
                st.markdown(create_download_link_png(png_buffer, "joint_plot.png", "Download Joint Plot as PNG"), unsafe_allow_html=True)

            # Violin plot
            if st.checkbox("Show Violin Plot"):
                st.write("Violin plots provide a deeper understanding of the distribution as compared to box plots.")
                
                # Using a unique key for this selectbox
                selected_cat = st.selectbox("Select categorical column for Violin Plot", cat_columns, key='violin_plot_select_cat')
                
                # Using a unique key for this selectbox
                selected_num = st.selectbox("Select numerical column for Violin Plot", num_columns, key='violin_plot_select_num')
                
                fig, ax = plt.subplots()
                sns.violinplot(data=filtered_data, x=selected_cat, y=selected_num, ax=ax)
                st.pyplot(fig)

                # Save the plot and generate a download link
                png_buffer = save_plot_to_png(fig)
                st.markdown(create_download_link_png(png_buffer, "violin_plot.png", "Download Violin Plot as PNG"), unsafe_allow_html=True)


        # Multivariate Analysis
        elif analysis_section == "Multivariate Analysis":
            st.subheader("Multivariate Analysis")
            
            # Select variables of interest
            st.write("Always Select ReportDate as one of the variables of interest")
            variables_of_interest = st.multiselect("Select variables for multivariate analysis", num_columns + cat_columns)

            if len(variables_of_interest) > 2:
                if st.checkbox("Show Pair Plot"):
                    # Assign a value to hue_column
                    hue_column = cat_columns[0] if cat_columns[0] in filtered_data.columns else cat_columns[1]

                    # Now you can safely check if hue_column exists in filtered_data
                    if hue_column not in filtered_data.columns:
                        st.error(f"'{hue_column}' column not found in the data.")
                        return

                    fig = sns.pairplot(filtered_data[variables_of_interest], hue=hue_column)
                    st.pyplot(fig.fig)

                    # Save the plot and generate a download link
                    png_buffer = save_plot_to_png(fig)
                    st.markdown(create_download_link_png(png_buffer, "pair_plot.png", "Download Pair Plot as PNG"), unsafe_allow_html=True)


                if st.checkbox("Show Correlation Heatmap"):
                    corr = filtered_data[variables_of_interest].corr()
                    fig, ax = plt.subplots(figsize=(10,7))
                    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                    st.pyplot(fig)

                    # Save the plot and generate a download link
                    png_buffer = save_plot_to_png(fig)
                    st.markdown(create_download_link_png(png_buffer, "heat_map.png", "Download Heatmap as PNG"), unsafe_allow_html=True)

                if st.checkbox("Show 3D Scatter Plot"):
                    x_var = st.selectbox("Select variable for X-axis", variables_of_interest)
                    y_var = st.selectbox("Select variable for Y-axis", variables_of_interest, index=1 if len(variables_of_interest) > 1 else 0)  # Choose the second variable by default
                    z_var = st.selectbox("Select variable for Z-axis", variables_of_interest, index=2 if len(variables_of_interest) > 2 else 0)  # Choose the third variable by default
                    
                    fig = px.scatter_3d(filtered_data, x=x_var, y=y_var, z=z_var, color=cat_columns[0])
                    st.plotly_chart(fig)
                    # st.write("Use the toolbar above to download the plot as PNG.")
                
                # Pair Grid
                if st.checkbox("Show Pair Grid"):
                    st.write("Pair Grid allows you to map different plots to the rows and columns of a grid, helping in studying multiple pairwise relationships.")
                    variables_for_pairgrid = st.multiselect("Select variables for Pair Grid", num_columns + cat_columns, default=num_columns)
                    hue_column = cat_columns[0] if cat_columns[0] in filtered_data.columns else cat_columns[1]  # Fallback to the second categorical column if the first isn't present
                    g = sns.PairGrid(filtered_data[variables_for_pairgrid], hue=hue_column)
                    g.map_upper(sns.scatterplot)
                    g.map_lower(sns.kdeplot)
                    g.map_diag(sns.histplot)
                    st.pyplot(g.fig)

                    # Save the plot and generate a download link
                    png_buffer = save_plot_to_png(fig)
                    st.markdown(create_download_link_png(png_buffer, "pair_grid.png", "Download Pair Grid Plot as PNG"), unsafe_allow_html=True)

                # Heatmap of Pivot Table
                if st.checkbox("Show Heatmap of Pivot Table"):
                    st.write("Visualize interactions between two categorical variables and one numerical variable using a heatmap.")
                    cat_col1 = st.selectbox("Select first categorical column", cat_columns)
                    cat_col2 = st.selectbox("Select second categorical column", cat_columns, index=1)
                    num_col_for_pivot = st.selectbox("Select numerical column for heatmap", num_columns)
                    pivot_table_result = pd.pivot_table(filtered_data, values=num_col_for_pivot, index=cat_col1, columns=cat_col2, aggfunc=np.mean)
                    fig, ax = plt.subplots()
                    sns.heatmap(pivot_table_result, annot=True, cmap='coolwarm', ax=ax)
                    st.pyplot(fig)

                    # Save the plot and generate a download link
                    png_buffer = save_plot_to_png(fig)
                    st.markdown(create_download_link_png(png_buffer, "pivot_heatmap.png", "Download Pivot Table Heatmap as PNG"), unsafe_allow_html=True)

                # Facet Grid
                if st.checkbox("Show Facet Grid"):
                    st.write("Visualize the distribution of one variable as well as the relationship between multiple variables, separately within subsets of your dataset.")
                    num_col1 = st.selectbox("Select first numerical column", num_columns)
                    num_col2 = st.selectbox("Select second numerical column", num_columns, index=1)
                    cat_col_for_facet = st.selectbox("Select categorical column for facet", cat_columns)
                    g = sns.FacetGrid(filtered_data, col=cat_col_for_facet, col_wrap=4, height=3)
                    g.map(sns.scatterplot, num_col1, num_col2, edgecolor="w")
                    st.pyplot(g.fig)

                    # Save the plot and generate a download link
                    png_buffer = save_plot_to_png(fig)
                    st.markdown(create_download_link_png(png_buffer, "facetgrid_plot.png", "Download Facet Grid Plot as PNG"), unsafe_allow_html=True)

                # Bubble Plot
                if st.checkbox("Show Bubble Plot"):
                    st.write("Bubble Plot is a scatter plot variation, where data points are replaced with bubbles. The size and color of the bubble can represent additional variables.")
                    x_var = st.selectbox("Select variable for X-axis", num_columns)
                    y_var = st.selectbox("Select variable for Y-axis", num_columns, index=1 if len(num_columns) > 1 else 0)
                    z_var = st.selectbox("Select variable for bubble size", num_columns, index=2 if len(num_columns) > 2 else 0)
                    color_var = st.selectbox("Select categorical variable for bubble color", cat_columns)
                    fig = px.scatter(filtered_data, x=x_var, y=y_var, size=z_var, color=color_var, hover_name="FullName", size_max=60)
                    st.plotly_chart(fig)
                    # st.write("Use the toolbar above to download the plot as PNG.")

            else:
                st.warning("Please select more than two variables for multivariate analysis.")
    

    # Add this line at the end of your main() function
    st.markdown("<small>Developed by Richard Kabiru</small>", unsafe_allow_html=True)
         


if __name__ == "__main__":
    main()
